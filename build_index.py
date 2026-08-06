"""
build_index.py

OFFLINE indexing job. Run this locally or on your HPC/GPU box, NOT inside the
deployed container. It parses data/*.pdf, chunks them at two granularities,
embeds them with bge-large-en-v1.5, and persists two Chroma stores to disk:

    chroma_db/        <- 400-token chunks
    chroma_db_large/  <- 1200-token chunks

The deployed FastAPI app (main.py) only ever LOADS these directories — it
never rebuilds them at container startup. This keeps the expensive,
GPU-friendly indexing step separate from the lightweight, CPU-friendly
serving step, which is the whole point of splitting this out.

Usage:
    python build_index.py                # build both stores from scratch
    python build_index.py --force        # wipe and rebuild even if they exist

After running, ship chroma_db/ and chroma_db_large/ to wherever the
deployed container reads its data from (Docker volume, k8s PVC, etc).
"""

import argparse
import logging
import os
import shutil
import time

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from device_utils import get_embedding_batch_size, get_embedding_device

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_index")

DATA_DIR = "data/"
SMALL_DB_PATH = "chroma_db"
LARGE_DB_PATH = "chroma_db_large"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def load_and_split_docs():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' is missing or empty. Add your source PDFs there before indexing."
        )

    logger.info("Loading PDFs with PyMuPDF...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDF pages parsed from '{DATA_DIR}'. Check the files there.")
    logger.info(f"Parsed {len(docs)} pages from {DATA_DIR}")

    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=80, add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    large_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=300, add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    small_chunks = small_splitter.split_documents(docs)
    large_chunks = large_splitter.split_documents(docs)
    logger.info(f"Small chunks (400): {len(small_chunks)} | Large chunks (1200): {len(large_chunks)}")
    return small_chunks, large_chunks


def build_store(chunks, persist_dir, embedding_function, force: bool):
    if force and os.path.exists(persist_dir):
        logger.info(f"--force set: removing existing store at {persist_dir}")
        shutil.rmtree(persist_dir)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info(f"Store already exists at {persist_dir}, skipping (use --force to rebuild).")
        return

    logger.info(f"Embedding {len(chunks)} chunks into {persist_dir}...")
    t0 = time.time()
    Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=persist_dir)
    logger.info(f"Done in {time.time() - t0:.1f}s -> {persist_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build the offline Chroma indexes for the RAG engine.")
    parser.add_argument("--force", action="store_true", help="Rebuild stores even if they already exist.")
    args = parser.parse_args()

    device = get_embedding_device()
    batch_size = get_embedding_batch_size(device if isinstance(device, str) else "gpu")
    logger.info(f"Embedding device: {device} | batch size: {batch_size}")

    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )

    small_chunks, large_chunks = load_and_split_docs()

    build_store(small_chunks, SMALL_DB_PATH, embedding_function, args.force)
    build_store(large_chunks, LARGE_DB_PATH, embedding_function, args.force)

    logger.info("Indexing complete. Ship chroma_db/ and chroma_db_large/ to your deployment target.")


if __name__ == "__main__":
    main()