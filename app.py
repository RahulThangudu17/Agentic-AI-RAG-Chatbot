import os
from typing import TypedDict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from pinecone import Pinecone
from langgraph.graph import StateGraph, START, END

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25"))

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Create a .env file from .env.example.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [item.name for item in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    raise RuntimeError(
        f"Pinecone index '{PINECONE_INDEX_NAME}' was not found. Run 'python ingest.py' first."
    )

index = pc.Index(PINECONE_INDEX_NAME)
index_info = pc.describe_index(PINECONE_INDEX_NAME)
if index_info.dimension != 384:
    raise RuntimeError(
        f"Pinecone index dimension is {index_info.dimension}; expected 384 for all-MiniLM-L6-v2."
    )

vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

NOT_FOUND = "I could not find this information in the provided Agentic AI eBook."

SYSTEM_PROMPT = """
You are an AI assistant that answers questions ONLY from the provided Agentic AI eBook context.

Rules:
1. Use only the retrieved context below.
2. Do not use outside knowledge, even if you know the answer.
3. Do not invent facts or make assumptions.
4. If the context does not contain enough information to answer the question, respond exactly:
   "I could not find this information in the provided Agentic AI eBook."
5. Keep the answer clear and concise.
6. Every factual statement must be supported by the retrieved context.

Retrieved context:
{context}

User question:
{question}
"""


class RAGState(TypedDict):
    question: str
    context: List[dict]
    answer: str
    confidence: float


def retrieve_node(state: RAGState):
    question = state["question"]
    results = vectorstore.similarity_search_with_score(question, k=RETRIEVAL_K)

    context = []
    seen = set()
    for doc, score in results:
        score = float(score)
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        if score < MIN_RETRIEVAL_SCORE:
            continue

        context.append({
            "text": doc.page_content,
            "page": int(doc.metadata.get("page", -1)) + 1,
            "score": round(score, 4),
        })
        if len(context) >= TOP_K:
            break

    confidence = (
        round(sum(item["score"] for item in context) / len(context), 4)
        if context else 0.0
    )

    return {"context": context, "confidence": confidence}


def generate_node(state: RAGState):
    context = state["context"]
    if not context:
        return {"answer": NOT_FOUND}

    context_text = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}" for item in context
    )
    prompt = SYSTEM_PROMPT.format(
        context=context_text,
        question=state["question"],
    )
    response = llm.invoke(prompt)
    answer = response.content.strip()

    return {"answer": answer or NOT_FOUND}


graph_builder = StateGraph(RAGState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
rag_graph = graph_builder.compile()


def ask_rag(question: str):
    result = rag_graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "confidence": 0.0,
    })
    return {
        "answer": result["answer"],
        "context": result["context"],
        "confidence": result["confidence"],
    }


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot grounded in the Agentic AI eBook",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the Agentic AI eBook")


@app.get("/")
def root():
    return {"message": "Agentic AI RAG Chatbot is running. Open /docs to test it."}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "Agentic AI RAG Chatbot"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return ask_rag(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc
