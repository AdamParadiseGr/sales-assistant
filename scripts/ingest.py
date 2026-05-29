#!/usr/bin/env python3
"""
Ingests knowledge base Markdown files into ChromaDB.

Usage:
    python scripts/ingest.py              # use .env settings
    python scripts/ingest.py --reset      # drop existing collection first
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

KB_DIR = ROOT / "app" / "data" / "knowledge_base"
CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(ROOT / "chroma_db"))
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents():
    from langchain_community.document_loaders import TextLoader

    docs = []
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        logger.error("No .md files found in %s", KB_DIR)
        sys.exit(1)

    for md_file in md_files:
        loader = TextLoader(str(md_file), encoding="utf-8")
        file_docs = loader.load()
        for doc in file_docs:
            doc.metadata["source"] = md_file.stem
            doc.metadata["filename"] = md_file.name
        docs.extend(file_docs)
        logger.info("Loaded: %s (%d chars)", md_file.name, sum(len(d.page_content) for d in file_docs))

    return docs


def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def build_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model: all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest(reset: bool = False) -> None:
    from langchain_chroma import Chroma

    logger.info("Knowledge base dir: %s", KB_DIR)
    logger.info("ChromaDB dir:       %s", CHROMA_DIR)

    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = build_embeddings()

    if reset:
        import shutil
        chroma_path = Path(CHROMA_DIR)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            logger.info("Existing ChromaDB deleted")

    logger.info("Ingesting %d chunks into ChromaDB...", len(chunks))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    # Smoke-test: verify a retrieval works
    results = vectorstore.similarity_search("тарифы РКО", k=3)
    logger.info("Smoke-test query 'тарифы РКО' returned %d results:", len(results))
    for i, r in enumerate(results, 1):
        logger.info("  %d. [%s] %s…", i, r.metadata.get("source"), r.page_content[:80])

    logger.info("Done. %d chunks stored in collection '%s'.", len(chunks), COLLECTION_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB collection before ingesting",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
