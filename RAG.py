from dotenv import load_dotenv
import os
import weaviate
from collections import Counter
import re
from uuid import uuid4
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.schema import Document

# Weaviate imports
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
weaviate_cluster_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate
auth_config = AuthApiKey(api_key=weaviate_api_key)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_cluster_url,
    auth_credentials=auth_config,
    headers={"X-OpenAI-Api-Key": openai_api_key}
)
print("Connected to Weaviate Cloud")

# Load PDFs with source metadata
pdf_files = ["Chunking_RAG1.pdf", "final_report_Harshit.pdf"]

pages = []
for file_path in pdf_files:
    print(f"Loading PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    loaded_pages = loader.load()
    for page in loaded_pages:
        if page.metadata is None:
            page.metadata = {}
        page.metadata["source_file"] = file_path
    pages.extend(loaded_pages)

print(f"Total pages loaded: {len(pages)}")

# Check page count per PDF
pdf_counts = {}
for p in pages:
    src = p.metadata.get("source_file", "unknown")
    pdf_counts[src] = pdf_counts.get(src, 0) + 1
print("✅ Page count per PDF:", pdf_counts)

# Custom Paragraph Splitter
class ParagraphSplitter:
    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            paragraphs = doc.page_content.split('\n\n')
            for para in paragraphs:
                cleaned = para.strip()
                if cleaned:
                    chunks.append(Document(page_content=cleaned, metadata=doc.metadata))
        return chunks

# Custom Keyword Splitter
class KeywordSplitter:
    def __init__(self, keywords=None):
        self.keywords = keywords or ["important", "summary", "conclusion", "note"]

    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            paragraphs = doc.page_content.split('\n\n')
            current_chunk = []
            for para in paragraphs:
                cleaned = para.strip()
                if not cleaned:
                    continue
                if any(keyword.lower() in cleaned.lower() for keyword in self.keywords):
                    if current_chunk:
                        chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
                        current_chunk = []
                    chunks.append(Document(page_content=cleaned, metadata=doc.metadata))
                else:
                    current_chunk.append(cleaned)
            if current_chunk:
                chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
        return chunks

# Table Splitter
class TableSplitter:
    def split_documents(self, docs):
        chunks = []
        table_pattern = re.compile(
            r"((\|.*\|[\r\n]+)+)",  # Matches markdown-style tables
            re.MULTILINE
        )
        for doc in docs:
            content = doc.page_content
            last_end = 0
            for match in table_pattern.finditer(content):
                start, end = match.span()
                if start > last_end:
                    pre_text = content[last_end:start].strip()
                    if pre_text:
                        chunks.append(Document(page_content=pre_text, metadata=doc.metadata))
                table_text = match.group().strip()
                if table_text:
                    chunks.append(Document(page_content=table_text, metadata=doc.metadata))
                last_end = end
            tail_text = content[last_end:].strip()
            if tail_text:
                chunks.append(Document(page_content=tail_text, metadata=doc.metadata))
        return chunks

# Topic-based Chunker (using simple keyword/topic grouping)
class TopicSplitter:
    def __init__(self, topics=None):
        # Example topics; customize with your own keywords for topics
        self.topics = topics or {
            "finance": ["finance", "budget", "investment", "money"],
            "technology": ["technology", "software", "hardware", "AI", "machine learning"],
            "health": ["health", "medicine", "wellness", "disease"],
        }

    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            current_topic = None
            current_chunk = []
            for para in paragraphs:
                para_lower = para.lower()
                matched_topic = None
                for topic, keywords in self.topics.items():
                    if any(keyword in para_lower for keyword in keywords):
                        matched_topic = topic
                        break
                if matched_topic != current_topic:
                    # Save previous chunk
                    if current_chunk:
                        chunks.append(Document(page_content="\n\n".join(current_chunk), metadata={**doc.metadata, "topic": current_topic or "unknown"}))
                        current_chunk = []
                    current_topic = matched_topic
                current_chunk.append(para)
            if current_chunk:
                chunks.append(Document(page_content="\n\n".join(current_chunk), metadata={**doc.metadata, "topic": current_topic or "unknown"}))
        return chunks

# Content-aware chunker (split on natural language cues)
class ContentAwareSplitter:
    def split_documents(self, docs):
        chunks = []
        sentence_endings = re.compile(r'(?<=[.!?]) +')
        for doc in docs:
            sentences = sentence_endings.split(doc.page_content)
            current_chunk = []
            current_len = 0
            max_chunk_size = 1000  # chars

            for sentence in sentences:
                s_len = len(sentence)
                if current_len + s_len > max_chunk_size and current_chunk:
                    chunks.append(Document(page_content=" ".join(current_chunk), metadata=doc.metadata))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(sentence)
                current_len += s_len
            if current_chunk:
                chunks.append(Document(page_content=" ".join(current_chunk), metadata=doc.metadata))
        return chunks

# Semantic chunker - cluster paragraphs based on embedding similarity (simple version)
class SemanticSplitter:
    def __init__(self, embedding_model, similarity_threshold=0.75):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            if not paragraphs:
                continue

            embeddings = self.embedding_model.embed_documents(paragraphs)
            embeddings = np.array(embeddings)

            current_chunk = []
            current_embeds = []
            for i, para in enumerate(paragraphs):
                para_embed = embeddings[i]

                if current_chunk:
                    mean_emb = np.mean(current_embeds, axis=0)
                    sim = cosine_similarity([mean_emb], [para_embed])[0][0]
                else:
                    sim = 1.0

                if sim >= self.similarity_threshold:
                    current_chunk.append(para)
                    current_embeds.append(para_embed)
                else:
                    chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
                    current_chunk = [para]
                    current_embeds = [para_embed]

            if current_chunk:
                chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
        return chunks

# Embedding chunker with chunk size and similarity threshold
class EmbeddingChunker:
    def __init__(self, embedding_model, max_chunk_size=1000, similarity_threshold=0.8):
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            if not paragraphs:
                continue

            embeddings = self.embedding_model.embed_documents(paragraphs)
            embeddings = np.array(embeddings)

            current_chunk = []
            current_embeds = []
            current_length = 0

            def mean_embedding(embeds):
                return np.mean(embeds, axis=0) if embeds else None

            for i, para in enumerate(paragraphs):
                para_len = len(para)
                para_embed = embeddings[i]

                if current_chunk:
                    mean_emb = mean_embedding(current_embeds)
                    sim = cosine_similarity([mean_emb], [para_embed])[0][0]
                else:
                    sim = 1.0

                if sim >= self.similarity_threshold and (current_length + para_len) <= self.max_chunk_size:
                    current_chunk.append(para)
                    current_embeds.append(para_embed)
                    current_length += para_len
                else:
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
                    current_chunk = [para]
                    current_embeds = [para_embed]
                    current_length = para_len

            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))

        return chunks

# Your existing Recursive, Fixed, Sliding, Sentence chunkers:
large_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
medium_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
small_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)

strategies = {
    "Recursive": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    "Fixed": CharacterTextSplitter(chunk_size=500, chunk_overlap=0),
    "Sliding": CharacterTextSplitter(chunk_size=500, chunk_overlap=250),
    "Sentence": TokenTextSplitter(chunk_size=500, chunk_overlap=50),
    "Paragraph": ParagraphSplitter(),
    "Keyword": KeywordSplitter(keywords=["important", "summary", "conclusion", "note"]),
    "Table": TableSplitter(),
    "Topic": TopicSplitter(),
    "ContentAware": ContentAwareSplitter(),
    "Semantic": SemanticSplitter(OpenAIEmbeddings(openai_api_key=openai_api_key), similarity_threshold=0.75),
    "EmbeddingChunking": EmbeddingChunker(OpenAIEmbeddings(openai_api_key=openai_api_key), max_chunk_size=1000, similarity_threshold=0.8),
}

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

query = input("🔍 Enter your query: ")

def run_strategy(name, splitter_or_func, query):
    print(f"\n--- Running chunking strategy: {name} ---")

    if callable(splitter_or_func):
        documents = splitter_or_func(pages)
    else:
        documents = splitter_or_func.split_documents(pages)

    chunk_counts = Counter(doc.metadata.get("source_file", "unknown") for doc in documents)
    print("📦 Chunk counts per PDF before indexing:", dict(chunk_counts))
    print(f"Total chunks generated: {len(documents)}")

    collection_name = f"Doc_Combined_{name}"
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    properties = [Property(name="text", data_type=DataType.TEXT)]

    client.collections.create(
        name=collection_name,
        properties=properties,
        vector_index_config=Configure.VectorIndex.hnsw(),
        vectorizer_config=Configure.Vectorizer.none()
    )

    documents_with_ids = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "uuid": doc.metadata.get("uuid", str(uuid4()))}
        )
        for doc in documents
    ]

    vectorstore = WeaviateVectorStore.from_documents(
        documents=documents_with_ids,
        embedding=embeddings,
        client=client,
        index_name=collection_name,
        text_key="text",
        by_text=False
    )

    failed = client.batch.failed_objects
    if failed:
        print(f"⚠️ Warning: {len(failed)} objects failed to index for strategy {name}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})

    print(f"\nStrategy: {name} — Answer:")
    print(result['result'])
    print(f"Chunks used in answer: {len(result['source_documents'])} / {len(documents)}")

    sources = [doc.metadata.get("source_file", "unknown") for doc in result['source_documents']]
    print("📄 PDFs contributing to answer:", set(sources))

for strat_name, splitter in strategies.items():
    run_strategy(strat_name, splitter, query)

client.close()
print("All done.")
