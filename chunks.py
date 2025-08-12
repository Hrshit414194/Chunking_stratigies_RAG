from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4

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

class TableSplitter:
    def split_documents(self, docs):
        chunks = []
        table_pattern = re.compile(r"((\|.*\|[\r\n]+)+)", re.MULTILINE)
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

class TopicSplitter:
    def __init__(self, topics=None):
        self.topics = topics or {
            "finance": ["finance", "budget", "investment", "money"],
            "technology": ["technology", "software", "hardware", "ai", "machine learning"],
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
                    if current_chunk:
                        chunks.append(Document(page_content="\n\n".join(current_chunk), metadata={**doc.metadata, "topic": current_topic or "unknown"}))
                        current_chunk = []
                    current_topic = matched_topic
                current_chunk.append(para)
            if current_chunk:
                chunks.append(Document(page_content="\n\n".join(current_chunk), metadata={**doc.metadata, "topic": current_topic or "unknown"}))
        return chunks

class ContentAwareSplitter:
    def split_documents(self, docs):
        chunks = []
        sentence_endings = re.compile(r'(?<=[.!?]) +')
        for doc in docs:
            sentences = sentence_endings.split(doc.page_content)
            current_chunk = []
            current_len = 0
            max_chunk_size = 1000

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

# Semantic & Embedding chunkers depend on an embedding model with embed_documents()
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

# Helpers to expose common splitters
def get_default_strategies(embeddings=None):
    # embedding-dependent strategies require embeddings to be passed
    strategies = {
        "Recursive": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
        "Fixed": CharacterTextSplitter(chunk_size=500, chunk_overlap=0),
        "Sliding": CharacterTextSplitter(chunk_size=500, chunk_overlap=250),
        "Sentence": TokenTextSplitter(chunk_size=500, chunk_overlap=50),
        "Paragraph": ParagraphSplitter(),
        "Keyword": KeywordSplitter(),
        "Table": TableSplitter(),
        "Topic": TopicSplitter(),
        "ContentAware": ContentAwareSplitter(),
    }
    if embeddings is not None:
        strategies["Semantic"] = SemanticSplitter(embeddings, similarity_threshold=0.75)
        strategies["EmbeddingChunking"] = EmbeddingChunker(embeddings, max_chunk_size=1000, similarity_threshold=0.8)
    return strategies
