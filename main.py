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


# import os
# import logging
# import traceback
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_weaviate import WeaviateVectorStore
# from langchain.chains import RetrievalQA
# from uuid import uuid4

# from chunks import get_default_strategies
# from db_utils import EmbeddingCache, WeaviateManager

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def load_pdfs(file_paths):
#     pages = []
#     for file_path in file_paths:
#         logger.info("Loading PDF: %s", file_path)
#         loader = PyPDFLoader(file_path)
#         try:
#             loaded_pages = loader.load()
#         except Exception as e:
#             logger.exception("Failed to load %s", file_path)
#             continue
#         for page in loaded_pages:
#             if page.metadata is None:
#                 page.metadata = {}
#             page.metadata["source_file"] = file_path
#             page.metadata.setdefault("uuid", str(uuid4()))
#         pages.extend(loaded_pages)
#     return pages


# def main():
#     try:
#         load_dotenv()
#         OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#         WEAVIATE_URL = os.getenv("WEAVIATE_URL")
#         WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

#         if not OPENAI_API_KEY:
#             raise RuntimeError("OPENAI_API_KEY not set")

#         # instantiate embedding model and LLM
#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#         llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

#         # create strategies (pass embeddings for semantic strategies)
#         strategies = get_default_strategies(embeddings=embeddings)

#         # load PDFs (customize)
#         pdf_files = ["Chunking_RAG1.pdf", "final_report_Harshit.pdf"]
#         pages = load_pdfs(pdf_files)
#         logger.info("Total pages loaded: %d", len(pages))

#         # Weaviate client
#         from weaviate.auth import AuthApiKey
#         import weaviate
#         auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)
#         client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=auth_config, headers={"X-OpenAI-Api-Key": OPENAI_API_KEY})
#         wm = WeaviateManager(client)

#         # embedding cache
#         cache = EmbeddingCache()

#         # pick a query from user
#         query = input("Enter your query: ")

#         for name, splitter in strategies.items():
#             try:
#                 logger.info("Running strategy %s", name)
#                 if hasattr(splitter, 'split_documents'):
#                     documents = splitter.split_documents(pages)
#                 elif callable(splitter):
#                     documents = splitter(pages)
#                 else:
#                     raise RuntimeError("Splitter for %s is unusable" % name)

#                 logger.info("Generated %d chunks", len(documents))

#                 collection_name = f"Doc_Combined_{name}"
#                 wm.create_collection(collection_name)

#                 # ensure each doc has uuid
#                 for doc in documents:
#                     doc.metadata.setdefault("uuid", str(uuid4()))

#                 vectorstore = WeaviateVectorStore.from_documents(
#                     documents=documents,
#                     embedding=embeddings,
#                     client=client,
#                     index_name=collection_name,
#                     text_key="text",
#                     by_text=False,
#                 )

#                 retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#                 qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
#                 result = qa_chain.invoke({"query": query})

#                 print(f"\nStrategy: {name} — Answer:")
#                 print(result['result'])
#                 print(f"Chunks used in answer: {len(result['source_documents'])} / {len(documents)}")

#             except Exception as e:
#                 logger.exception("Strategy %s failed: %s", name, e)
#                 # continue with other strategies

#         client.close()
#         logger.info("All done")

#     except Exception as e:
#         logger.exception("Pipeline failed: %s", e)
#         print("Pipeline failed. Check logs for details.")

# if __name__ == '__main__':
#     main()
