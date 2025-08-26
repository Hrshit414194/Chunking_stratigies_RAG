from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from chunks import Chunker
from db_utils import run_strategy, embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter


# Load PDFs

pdf_files = ["Chunking_RAG1.pdf", "final_report_Harshit.pdf"]
pages = []

for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    loaded_pages = loader.load()
    for page in loaded_pages:
        if page.metadata is None:
            page.metadata = {}
        page.metadata["source_file"] = file_path
    pages.extend(loaded_pages)

print(f"✅ Total pages loaded: {len(pages)}")

# Initialize single Chunker object

chunker = Chunker(embedding_model=embeddings, similarity_threshold=0.75, max_chunk_size=1000)


# Define strategies

strategies = {
    "Recursive": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    "Fixed": CharacterTextSplitter(chunk_size=500, chunk_overlap=0),
    "Sliding": CharacterTextSplitter(chunk_size=500, chunk_overlap=250),
    "Sentence": TokenTextSplitter(chunk_size=500, chunk_overlap=50),
    "Paragraph": chunker.paragraph_split,
    "Keyword": chunker.keyword_split,
    "Table": chunker.table_split,
    "Topic": chunker.topic_split,
    "ContentAware": chunker.content_aware_split,
    "Semantic": chunker.semantic_split,
    "EmbeddingChunking": chunker.embedding_split,
}


# FastAPI app

app = FastAPI(title="RAG PDF Chatbot API")

class QueryRequest(BaseModel):
    query: str
    strategy: Optional[str] = "Recursive"

class QueryResponse(BaseModel):
    answer: str
    chunks_used: int
    total_chunks: int
    sources: List[str]

@app.get("/")
def root():
    return {"message": "Harshit's RAG PDF Chatbot API is running. Use /strategies or /query"}

@app.get("/strategies")
def get_strategies():
    return {"available_strategies": list(strategies.keys())}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    strategy_name = request.strategy
    query_text = request.query

    if strategy_name not in strategies:
        raise HTTPException(status_code=400, detail=f"Strategy '{strategy_name}' not found.")

    try:
        # Run the selected strategy
        result, total_chunks = run_strategy(strategy_name, strategies[strategy_name], query_text, pages)
        chunks_used = len(result['source_documents'])
        sources = [doc.metadata.get("source_file", "unknown") for doc in result['source_documents']]

        return QueryResponse(
            answer=result['result'],
            chunks_used=chunks_used,
            total_chunks=total_chunks,
            sources=list(set(sources))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
