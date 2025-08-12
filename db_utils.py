import os
import logging
import shelve
from uuid import uuid4
import weaviate
from weaviate.classes.config import Configure, Property, DataType

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Simple disk-backed cache for embeddings using shelve."""
    def __init__(self, path="embedding_cache.db"):
        self.path = path

    def _key(self, text):
        # use short deterministic key; for long-term use consider hashing
        return str(hash(text))

    def get(self, text):
        with shelve.open(self.path) as db:
            return db.get(self._key(text))

    def set(self, text, embedding):
        with shelve.open(self.path) as db:
            db[self._key(text)] = embedding

class WeaviateManager:
    def __init__(self, client):
        self.client = client

    def create_collection(self, name):
        try:
            if self.client.collections.exists(name):
                logger.info("Collection %s exists. Deleting and recreating.", name)
                self.client.collections.delete(name)

            properties = [Property(name="text", data_type=DataType.TEXT)]
            self.client.collections.create(
                name=name,
                properties=properties,
                vector_index_config=Configure.VectorIndex.hnsw(),
                vectorizer_config=Configure.Vectorizer.none(),
            )
            logger.info("Created collection %s", name)
        except Exception as e:
            logger.exception("Failed to create collection %s: %s", name, e)
            raise

    def index_documents(self, vectorstore, docs):
        # Vectorstore (langchain_weaviate) will handle batch upload; we keep this wrapper
        try:
            vectorstore.from_documents(documents=docs)
        except Exception:
            logger.exception("Indexing documents failed.")
            # re-raise for caller to decide
            raise