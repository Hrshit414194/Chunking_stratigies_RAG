import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter


class Chunker:
    def __init__(self, embedding_model=None, similarity_threshold=0.75, max_chunk_size=1000):
        """
        :param embedding_model: Optional embedding model for semantic/embedding-based chunking
        :param similarity_threshold: Threshold for semantic similarity
        :param max_chunk_size: Max size of chunks for embedding-based splitting
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

   
    # 1. Paragraph Splitter
   
    def paragraph_split(self, docs):
        chunks = []
        for doc in docs:
            paragraphs = doc.page_content.split('\n\n')
            for para in paragraphs:
                cleaned = para.strip()
                if cleaned:
                    chunks.append(Document(page_content=cleaned, metadata=doc.metadata))
        return chunks


    # 2. Keyword Splitter
 
    def keyword_split(self, docs, keywords=None):
        keywords = keywords or ["important", "summary", "conclusion", "note"]
        chunks = []
        for doc in docs:
            paragraphs = doc.page_content.split('\n\n')
            current_chunk = []
            for para in paragraphs:
                cleaned = para.strip()
                if not cleaned:
                    continue
                if any(keyword.lower() in cleaned.lower() for keyword in keywords):
                    if current_chunk:
                        chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
                        current_chunk = []
                    chunks.append(Document(page_content=cleaned, metadata=doc.metadata))
                else:
                    current_chunk.append(cleaned)
            if current_chunk:
                chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
        return chunks

   
    # 3. Table Splitter
   
    def table_split(self, docs):
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

  
    # 4. Topic Splitter
   
    def topic_split(self, docs, topics=None):
        topics = topics or {
            "finance": ["finance", "budget", "investment", "money"],
            "technology": ["technology", "software", "hardware", "AI", "machine learning"],
            "health": ["health", "medicine", "wellness", "disease"],
        }
        chunks = []
        for doc in docs:
            paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            current_topic = None
            current_chunk = []
            for para in paragraphs:
                para_lower = para.lower()
                matched_topic = None
                for topic, keywords in topics.items():
                    if any(keyword in para_lower for keyword in keywords):
                        matched_topic = topic
                        break
                if matched_topic != current_topic:
                    if current_chunk:
                        chunks.append(Document(
                            page_content="\n\n".join(current_chunk),
                            metadata={**doc.metadata, "topic": current_topic or "unknown"}
                        ))
                        current_chunk = []
                    current_topic = matched_topic
                current_chunk.append(para)
            if current_chunk:
                chunks.append(Document(
                    page_content="\n\n".join(current_chunk),
                    metadata={**doc.metadata, "topic": current_topic or "unknown"}
                ))
        return chunks


    # 5. Content Aware Splitter

    def content_aware_split(self, docs):
        chunks = []
        sentence_endings = re.compile(r'(?<=[.!?]) +')
        for doc in docs:
            sentences = sentence_endings.split(doc.page_content)
            current_chunk = []
            current_len = 0
            for sentence in sentences:
                s_len = len(sentence)
                if current_len + s_len > self.max_chunk_size and current_chunk:
                    chunks.append(Document(page_content=" ".join(current_chunk), metadata=doc.metadata))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(sentence)
                current_len += s_len
            if current_chunk:
                chunks.append(Document(page_content=" ".join(current_chunk), metadata=doc.metadata))
        return chunks

 
    # 6. Semantic Splitter
    
    def semantic_split(self, docs):
        if not self.embedding_model:
            raise ValueError("Embedding model required for semantic split")
        
        chunks = []
        for doc in docs:
            paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            if not paragraphs:
                continue
            embeddings = self.embedding_model.embed_documents(paragraphs)
            embeddings = np.array(embeddings)

            current_chunk, current_embeds = [], []
            for i, para in enumerate(paragraphs):
                para_embed = embeddings[i]
                sim = cosine_similarity([np.mean(current_embeds, axis=0)], [para_embed])[0][0] if current_embeds else 1.0
                if sim >= self.similarity_threshold:
                    current_chunk.append(para)
                    current_embeds.append(para_embed)
                else:
                    chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
                    current_chunk, current_embeds = [para], [para_embed]
            if current_chunk:
                chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
        return chunks


    # 7. Embedding-Aware Splitter
    
    def embedding_split(self, docs):
        if not self.embedding_model:
            raise ValueError("Embedding model required for embedding split")

        chunks = []
        for doc in docs:
            paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
            if not paragraphs:
                continue

            embeddings = self.embedding_model.embed_documents(paragraphs)
            embeddings = np.array(embeddings)

            current_chunk, current_embeds, current_length = [], [], 0
            for i, para in enumerate(paragraphs):
                para_len = len(para)
                para_embed = embeddings[i]
                sim = cosine_similarity([np.mean(current_embeds, axis=0)], [para_embed])[0][0] if current_embeds else 1.0

                if sim >= self.similarity_threshold and (current_length + para_len) <= self.max_chunk_size:
                    current_chunk.append(para)
                    current_embeds.append(para_embed)
                    current_length += para_len
                else:
                    if current_chunk:
                        chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))
                    current_chunk, current_embeds, current_length = [para], [para_embed], para_len

            if current_chunk:
                chunks.append(Document(page_content="\n\n".join(current_chunk), metadata=doc.metadata))

        return chunks


