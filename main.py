import os
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated, List

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Or whichever embeddings you used
from dotenv import load_dotenv
import torch_directml
from langfuse.langchain import CallbackHandler

load_dotenv()

# ==========================================
# 1. API INITIALIZATION & MODELS
# ==========================================
app = FastAPI(
    title="ML Infrastructure RAG Engine",
    description="A production-grade, dual-retrieval RAG system for Deep Learning architecture analysis.",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    session_id: Annotated[str, Field(
        description="A unique UUID string identifying the user's session.",
        examples=["user-123e4567-e89b-12d3-a456-426614174000"]
    )]
    query: Annotated[str, Field(
        description="The natural language question to ask the RAG engine.",
        examples=["What is the parameter size of Llama 3?"]
    )]

class ChatResponse(BaseModel):
    answer: str
    citations: List[str]

# ==========================================
# RETRIEVER DEFINITIONS & DEDUPLICATION
# ==========================================
def get_char_span(doc: Document):
    source = doc.metadata.get('source', '')
    start  = doc.metadata.get('start_index', None)
    end    = start + len(doc.page_content) if start is not None else None
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
        deduped = deduplicate_by_span(merged, threshold=self.dedup_threshold)
        return deduped
    
# ==========================================
# 2. STARTUP EVENTS (Loading the Brain)
# ==========================================
@app.on_event("startup")
async def load_resources():
    """Loads models, prompts, documents, and vector databases into server memory."""
    print("[SYSTEM] Booting up ML Infrastructure RAG Engine...")
    
    # 1. Load Prompts
    with open("prompts.yaml", "r") as file:
        app.state.config = yaml.safe_load(file)
        
    # 2. Initialize LLM
    app.state.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=2048)
    
    # 3. Build Chains
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", app.state.config["query_rewriter"]["system"]),
        ("human", app.state.config["query_rewriter"]["human"])
    ])
    app.state.query_rewriter = rewrite_prompt | app.state.llm | StrOutputParser()
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", app.state.config["qa_generator"]["system"]),
        ("human", app.state.config["qa_generator"]["human"]),
    ])
    app.state.qa_chain = create_stuff_documents_chain(app.state.llm, qa_prompt)

    # 4. INITIALIZE YOUR RETRIEVERS HERE
    print("[SYSTEM] Loading documents with PyMuPDF...")
    
    # Failsafe: Ensure the data directory exists
    if not os.path.exists("data/"):
        os.makedirs("data/")
        print("[WARNING] Created 'data/' directory. Please ensure your PDFs are inside it.")

    loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    print(f"[SYSTEM] Successfully parsed {len(docs)} pages.")

    # --- Small chunks for precise retrieval ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=80,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[SYSTEM] Small chunks (400): {len(chunks)}")

    # --- Large chunks for context-rich retrieval ---
    large_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    large_chunks = large_text_splitter.split_documents(docs)
    print(f"[SYSTEM] Large chunks (1200): {len(large_chunks)}")

    print("[SYSTEM] Connecting to Vector Databases...")
    
    # Initialize BGE-Large-En-v1.5 Embedding Model on AMD GPU
    print("[SYSTEM] Initializing DirectML on AMD GPU for Embeddings...")
    model_kwargs = {'device': torch_directml.device()}
    encode_kwargs = {
        'normalize_embeddings': True,
        'batch_size': 4
    }

    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    app.state.embedding_function = embedding_function

    # --- CHROMA DATABASES ---
    small_db_path = "chroma_db"
    if os.path.exists(small_db_path) and os.listdir(small_db_path):
        print(f"[SYSTEM] Loading existing small-chunk DB from {small_db_path}...")
        vectorstore = Chroma(persist_directory=small_db_path, embedding_function=embedding_function)
    else:
        print(f"[SYSTEM] Building NEW small-chunk DB at {small_db_path}...")
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=small_db_path)

    large_db_path = "chroma_db_large"
    if os.path.exists(large_db_path) and os.listdir(large_db_path):
        print(f"[SYSTEM] Loading existing large-chunk DB from {large_db_path}...")
        large_vectorstore = Chroma(persist_directory=large_db_path, embedding_function=embedding_function)
    else:
        print(f"[SYSTEM] Building NEW large-chunk DB at {large_db_path}...")
        large_vectorstore = Chroma.from_documents(documents=large_chunks, embedding=embedding_function, persist_directory=large_db_path)

    # --- ADVANCED RAG ARCHITECTURE ---
    vector_retriever = vectorstore.as_retriever(
        search_type='mmr', search_kwargs={"k": 20, "fetch_k": 40,"lambda_mult": 0.8},
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20
    small_ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever])

    large_vector_retriever = large_vectorstore.as_retriever(
        search_type='mmr', search_kwargs={"k": 20, "fetch_k": 40,"lambda_mult": 0.8}
    )
    large_bm25_retriever = BM25Retriever.from_documents(large_chunks)
    large_bm25_retriever.k = 20
    large_ensemble = EnsembleRetriever(retrievers=[large_bm25_retriever, large_vector_retriever])

    dual_retriever = DualEnsembleRetriever(
        small_retriever=small_ensemble,
        large_retriever=large_ensemble,
        dedup_threshold=0.7
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=10)

    # Attach the final pipeline to app.state
    app.state.retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=dual_retriever
    )
    
    print("[SYSTEM] Server Ready to accept requests.")
    
# ==========================================
# 3. THE CORE API ENDPOINT
# ==========================================
@app.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """The main Split-Brain execution pipeline with Langfuse Tracing."""
    
    if app.state.retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized.")

    try:
        print(f"\n[API] New Request from Session {request.session_id[-6:]}: {request.query}")
        
        # --- INITIALIZE LANGFUSE TRACKER ---
        # The new SDK requires an empty handler.
        langfuse_handler = CallbackHandler()
        
        # We now pass the session details dynamically via metadata to prevent memory leaks!
        langfuse_config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": request.session_id,
                "langfuse_user_id": "local_developer"
            }
        }

        # Step 1: Load User's Specific Memory
        history = SQLChatMessageHistory(
            session_id=request.session_id, 
            connection_string="sqlite:///chat_memory.db"
        )
        
        past_messages = history.messages[-4:]
        chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in past_messages])
        if not chat_history_str:
            chat_history_str = "No prior conversation."

        # Step 2: Smart Query Rewrite (Now Traced!)
        print("[API] Optimizing query...")
        optimized_query = app.state.query_rewriter.invoke(
            {"input": request.query, "chat_history": chat_history_str},
            config=langfuse_config  # <-- Attached here!
        )
        print(f"[API] Rewritten as: {optimized_query}")

        # Step 3: Dual-Retrieval & Reranking (Now Traced!)
        print("[API] Retrieving chunks...")
        retrieved_docs = app.state.retriever.invoke(
            optimized_query,
            config=langfuse_config  # <-- Attached here!
        )

        # Step 4: Final Answer Generation (Now Traced!)
        print("[API] Generating response...")
        answer = app.state.qa_chain.invoke(
            {"input": request.query, "context": retrieved_docs},
            config=langfuse_config  # <-- Attached here!
        )

       # Step 5: Format Citations (Fixing the 0-index bug)
        unique_citations = set()
        for doc in retrieved_docs: 
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            
            # Grab the page, default to 0 if missing, and add 1 for human readability
            raw_page = doc.metadata.get('page', 0)
            human_page = int(raw_page) + 1 
            
            unique_citations.add(f"{source} (Page {human_page})")
            
        citations = list(unique_citations)[:3] 

        # Step 6: Save to Memory
        history.add_user_message(request.query)
        history.add_ai_message(answer)
        
        print("[API] Request complete. Telemetry sent to Langfuse.")
        return ChatResponse(answer=answer, citations=citations)

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))