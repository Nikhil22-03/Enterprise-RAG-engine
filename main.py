import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, List

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from device_utils import get_embedding_batch_size, get_embedding_device

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rag_engine")

SMALL_DB_PATH = "chroma_db"
LARGE_DB_PATH = "chroma_db_large"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


# ==========================================
# RETRIEVER DEFINITIONS & DEDUPLICATION
# ==========================================
def get_char_span(doc: Document):
    source = doc.metadata.get("source", "")
    start = doc.metadata.get("start_index", None)
    end = start + len(doc.page_content) if start is not None else None
    return source, start, end


def overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    if None in (a_start, a_end, b_start, b_end):
        return 0.0
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    shorter = min(a_end - a_start, b_end - b_start)
    return overlap / shorter if shorter > 0 else 0.0


def deduplicate_by_span(docs: List[Document], threshold: float = 0.6) -> List[Document]:
    kept = []
    seen_hashes = set()
    for doc in docs:
        src, start, end = get_char_span(doc)
        if start is not None:
            duplicate = False
            for k in kept:
                k_src, k_start, k_end = get_char_span(k)
                if k_src == src and overlap_ratio(start, end, k_start, k_end) >= threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(doc)
        else:
            h = hash(doc.page_content.strip())
            if h not in seen_hashes:
                seen_hashes.add(h)
                kept.append(doc)
    return kept


class DualEnsembleRetriever(BaseRetriever):
    small_retriever: BaseRetriever
    large_retriever: BaseRetriever
    dedup_threshold: float = 0.6

    def _get_relevant_documents(self, query: str) -> List[Document]:
        small_docs = self.small_retriever.invoke(query)
        large_docs = self.large_retriever.invoke(query)
        merged = small_docs + large_docs
        return deduplicate_by_span(merged, threshold=self.dedup_threshold)


# ==========================================
# STARTUP: LOAD (NOT BUILD) THE INDEX
# ==========================================
def load_prebuilt_retriever():
    """
    Loads the two Chroma stores that build_index.py produced offline.
    Does NOT parse PDFs or embed anything at request-serving time —
    that expensive work happens once, offline, ideally on a GPU box.
    """
    for path in (SMALL_DB_PATH, LARGE_DB_PATH):
        if not (os.path.exists(path) and os.listdir(path)):
            raise RuntimeError(
                f"Missing pre-built index at '{path}'. Run `python build_index.py` "
                f"offline and ship the resulting directory before starting the server."
            )

    device = get_embedding_device()
    batch_size = get_embedding_batch_size(device if isinstance(device, str) else "gpu")
    logger.info(f"Embedding device for query-time inference: {device}")

    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )

    vectorstore = Chroma(persist_directory=SMALL_DB_PATH, embedding_function=embedding_function)
    large_vectorstore = Chroma(persist_directory=LARGE_DB_PATH, embedding_function=embedding_function)

    # BM25 needs the raw chunk text in memory; pull it back out of Chroma
    # rather than re-parsing PDFs, so this stays a "load" step, not a "build" step.
    small_chunks = vectorstore.get()
    small_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(small_chunks["documents"], small_chunks["metadatas"])
    ]
    large_chunks = large_vectorstore.get()
    large_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(large_chunks["documents"], large_chunks["metadatas"])
    ]

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 20, "fetch_k": 40, "lambda_mult": 0.8}
    )
    bm25_retriever = BM25Retriever.from_documents(small_docs)
    bm25_retriever.k = 20
    small_ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever])

    large_vector_retriever = large_vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 20, "fetch_k": 40, "lambda_mult": 0.8}
    )
    large_bm25_retriever = BM25Retriever.from_documents(large_docs)
    large_bm25_retriever.k = 20
    large_ensemble = EnsembleRetriever(retrievers=[large_bm25_retriever, large_vector_retriever])

    dual_retriever = DualEnsembleRetriever(
        small_retriever=small_ensemble, large_retriever=large_ensemble, dedup_threshold=0.7
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=10)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=dual_retriever)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting ML Infrastructure RAG Engine...")

    with open("prompts.yaml", "r") as file:
        app.state.config = yaml.safe_load(file)

    app.state.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=2048)

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", app.state.config["query_rewriter"]["system"]),
            ("human", app.state.config["query_rewriter"]["human"]),
        ]
    )
    app.state.query_rewriter = rewrite_prompt | app.state.llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", app.state.config["qa_generator"]["system"]),
            ("human", app.state.config["qa_generator"]["human"]),
        ]
    )
    app.state.qa_chain = create_stuff_documents_chain(app.state.llm, qa_prompt)

    app.state.retriever = None  # explicit default so /health never AttributeErrors
    app.state.retriever = load_prebuilt_retriever()

    logger.info("Server ready to accept requests.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="ML Infrastructure RAG Engine",
    description="A production-grade, dual-retrieval RAG system for Deep Learning architecture analysis.",
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================
# SCHEMAS
# ==========================================
class ChatRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(
            description="A unique UUID string identifying the user's session.",
            examples=["user-123e4567-e89b-12d3-a456-426614174000"],
        ),
    ]
    query: Annotated[
        str,
        Field(
            description="The natural language question to ask the RAG engine.",
            examples=["What is the parameter size of Llama 3?"],
        ),
    ]


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]


# ==========================================
# HEALTH CHECK (for k8s liveness/readiness probes)
# ==========================================
@app.get("/health")
def health():
    if getattr(app.state, "retriever", None) is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized.")
    return {"status": "ok"}


# ==========================================
# CORE ENDPOINT
# ==========================================
@app.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """The main Split-Brain execution pipeline with Langfuse tracing."""

    if getattr(app.state, "retriever", None) is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized.")

    try:
        logger.info(f"New request from session {request.session_id[-6:]}: {request.query}")

        langfuse_handler = CallbackHandler()
        langfuse_config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": request.session_id,
                "langfuse_user_id": "local_developer",
            },
        }

        history = SQLChatMessageHistory(
            session_id=request.session_id, connection_string="sqlite:///chat_memory.db"
        )

        past_messages = history.messages[-4:]
        chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in past_messages])
        if not chat_history_str:
            chat_history_str = "No prior conversation."

        logger.info("Optimizing query...")
        optimized_query = app.state.query_rewriter.invoke(
            {"input": request.query, "chat_history": chat_history_str}, config=langfuse_config
        )
        logger.info(f"Rewritten as: {optimized_query}")

        logger.info("Retrieving chunks...")
        retrieved_docs = app.state.retriever.invoke(optimized_query, config=langfuse_config)

        logger.info("Generating response...")
        answer = app.state.qa_chain.invoke(
            {"input": request.query, "context": retrieved_docs}, config=langfuse_config
        )

        # Order-preserving dedup: retrieved_docs is already ranked by the
        # cross-encoder, so we must not lose that order (a bare `set()` does).
        seen = set()
        citations: List[str] = []
        for doc in retrieved_docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            human_page = int(doc.metadata.get("page", 0)) + 1
            label = f"{source} (Page {human_page})"
            if label not in seen:
                seen.add(label)
                citations.append(label)
            if len(citations) == 3:
                break

        history.add_user_message(request.query)
        history.add_ai_message(answer)

        logger.info("Request complete. Telemetry sent to Langfuse.")
        return ChatResponse(answer=answer, citations=citations)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
