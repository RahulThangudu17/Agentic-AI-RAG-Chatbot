# Agentic AI RAG Chatbot

A Python RAG chatbot built for the **Appening Infotech AI Engineering Intern interview task**. The chatbot answers questions strictly from the provided Agentic AI eBook.

Knowledge base: `https://konverge.ai/pdf/Ebook-Agentic-AI.pdf`

## Architecture

```text
Agentic AI eBook PDF
        |
        v
   PyPDFLoader
        |
        v
RecursiveCharacterTextSplitter
        |
        v
Hugging Face Embeddings
(all-MiniLM-L6-v2)
        |
        v
   Pinecone Vector DB
        |
        | user question
        v
   LangGraph RAG
   +------------+
   | Retrieve   |
   +-----+------+
         |
         v
   +-----------+
   | Generate  |
   +-----+-----+
         |
         v
 Ollama / Llama 3.2
         |
         v
      FastAPI
       /chat
```

## What the API returns

Every successful `/chat` response contains:

- `answer`: final grounded answer
- `context`: retrieved chunks, page numbers, and Pinecone similarity scores
- `confidence`: average retrieval similarity score; **this is not a probability of answer correctness**

## Project structure

```text
agentic-ai-rag-chatbot/
|-- app.py                         # FastAPI + LangGraph RAG application
|-- ingest.py                      # PDF ingestion, chunking, embeddings, Pinecone storage
|-- agentic_ai_rag_chatbot.ipynb   # Notebook version/reference implementation
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## Prerequisites

Install:

1. Python 3.10+
2. Ollama
3. A Pinecone account and API key

Install Ollama from the official website, then download the model:

```bash
ollama pull llama3.2
```

Keep Ollama running while using the API.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd agentic-ai-rag-chatbot
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your Pinecone API key:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
OLLAMA_MODEL=llama3.2
TOP_K=5
RETRIEVAL_K=10
MIN_RETRIEVAL_SCORE=0.25
```

**Never commit the real `.env` file or API keys to GitHub.**

### 5. Ingest the knowledge base

Run:

```bash
python ingest.py
```

This script:

1. Downloads the Agentic AI eBook if it is not already present.
2. Loads the PDF with `PyPDFLoader`.
3. Splits the document into overlapping chunks.
4. Generates `all-MiniLM-L6-v2` embeddings.
5. Creates/connects to a Pinecone cosine-similarity index.
6. Stores the chunks and embeddings in Pinecone.

The script uses stable chunk IDs so ingestion is repeatable. It clears the configured project index before storing a fresh copy. Use a dedicated Pinecone index for this project.

### 6. Start the FastAPI server

```bash
ollama pull llama3.2
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API usage

### Health check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "service": "Agentic AI RAG Chatbot"
}
```

### Chat

```http
POST /chat
```

Request:

```json
{
  "question": "What is Agentic AI?"
}
```

Response shape:

```json
{
  "answer": "...",
  "context": [
    {
      "text": "...",
      "page": 8,
      "score": 0.1234
    }
  ],
  "confidence": 0.1234
}
```

### cURL example

```bash
curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is Agentic AI?\"}"
```

For PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -ContentType "application/json" -Body '{"question":"What is Agentic AI?"}'
```

## Sample queries

1. What is Agentic AI?
2. What are AI agents?
3. What are the characteristics of Agentic AI?
4. How do AI agents differ from traditional AI?
5. What are the components of an AI agent?
6. What are the applications of Agentic AI?

Out-of-scope test:

```text
Who is the current Prime Minister of India?
```

The expected grounded response for information not supported by the retrieved eBook context is:

```text
I could not find this information in the provided Agentic AI eBook.
```

## Grounding / hallucination control

The generation prompt tells the LLM to:

- use only retrieved eBook context;
- never use outside knowledge;
- never invent facts or make assumptions; and
- return the fallback response when the context is insufficient.

Retrieval also applies a configurable minimum similarity threshold (`MIN_RETRIEVAL_SCORE`). This prevents low-similarity results from being passed to the generation step.

The returned score is a **Pinecone cosine similarity score** and the reported `confidence` is the mean of the retained retrieval scores. It should not be interpreted as a calibrated probability.

## LangGraph workflow

The graph contains two nodes:

```text
START -> retrieve -> generate -> END
```

`retrieve` searches Pinecone and builds the context. `generate` sends only that context to the local Llama 3.2 model and produces the final answer.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Application implementation |
| LangChain | Document loading, splitting, embeddings, vector store and LLM integration |
| LangGraph | RAG workflow orchestration |
| Pinecone | Vector database / similarity search |
| Hugging Face | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Ollama | Local LLM runtime |
| Llama 3.2 | Answer generation |
| PyPDF | PDF ingestion |
| FastAPI | REST API |
| Uvicorn | ASGI server |

## Interview task mapping

| Requirement | Implementation |
|---|---|
| Ingest PDF | `ingest.py` + `PyPDFLoader` |
| Chunk text | `RecursiveCharacterTextSplitter` |
| Generate embeddings | `HuggingFaceEmbeddings` |
| Store in Pinecone | `PineconeVectorStore` |
| LangGraph RAG | `retrieve` + `generate` nodes |
| Grounded LLM | Ollama `llama3.2` + restricted prompt |
| API | FastAPI `/chat` |
| Final answer | `answer` field |
| Retrieved chunks | `context` field |
| Confidence/score | `confidence` + per-chunk `score` |
| Setup instructions | This README |
| Sample queries | Six queries above |
| Architecture | Diagram above |

## Security note

The repository intentionally does not contain the Pinecone API key. Create `.env` locally from `.env.example`; `.gitignore` excludes `.env`.
