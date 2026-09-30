import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag")
PDF_URL = os.getenv("PDF_URL", "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf")
PDF_PATH = Path(os.getenv("PDF_PATH", "Ebook-Agentic-AI.pdf"))

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Create a .env file from .env.example.")


def download_pdf():
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10_000:
        print(f"Using existing PDF: {PDF_PATH.resolve()}")
        return

    print("Downloading Agentic AI eBook...")
    response = requests.get(PDF_URL, timeout=60)
    response.raise_for_status()
    PDF_PATH.write_bytes(response.content)
    print(f"Downloaded: {PDF_PATH.resolve()}")


def main():
    download_pdf()

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    if not documents:
        raise RuntimeError("No pages were extracted from the PDF.")

    words = sum(len(doc.page_content.split()) for doc in documents)
    if words < 1000:
        raise RuntimeError("PDF text extraction returned too little text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_chunks = splitter.split_documents(documents)

    chunks = [
        chunk for chunk in all_chunks
        if len(chunk.page_content.split()) >= 25
    ]

    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)
    chunks = unique_chunks

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks after filtering: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    dimension = len(embeddings.embed_query("What is Agentic AI?"))
    if dimension != 384:
        raise RuntimeError(f"Unexpected embedding dimension: {dimension}")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [item.name for item in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Created Pinecone index.")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)

    index_info = pc.describe_index(PINECONE_INDEX_NAME)
    if index_info.dimension != dimension:
        raise RuntimeError(
            f"Index dimension is {index_info.dimension}, but embeddings are {dimension}. "
            "Use a new PINECONE_INDEX_NAME."
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    print("Clearing existing vectors from the project index...")
    index.delete(delete_all=True)
    time.sleep(2)

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    ids = [f"chunk-{i:04d}" for i in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)

    for _ in range(20):
        time.sleep(2)
        count = index.describe_index_stats().total_vector_count
        if count == len(chunks):
            break

    print(f"Vectors stored: {count}")
    if count != len(chunks):
        raise RuntimeError("Pinecone vector count does not match the chunk count yet.")

    print("Ingestion completed successfully.")
    print("Next step: start Ollama with 'ollama pull llama3.2', then run 'uvicorn app:app --reload'.")


if __name__ == "__main__":
    main()
