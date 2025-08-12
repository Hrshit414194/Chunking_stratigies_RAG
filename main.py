import os
import logging
import traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from langchain.chains import RetrievalQA
from uuid import uuid4

from chunks import get_default_strategies
from db_utils import EmbeddingCache, WeaviateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdfs(file_paths):
    pages = []
    for file_path in file_paths:
        logger.info("Loading PDF: %s", file_path)
        loader = PyPDFLoader(file_path)
        try:
            loaded_pages = loader.load()
        except Exception as e:
            logger.exception("Failed to load %s", file_path)
            continue
        for page in loaded_pages:
            if page.metadata is None:
                page.metadata = {}
            page.metadata["source_file"] = file_path
            page.metadata.setdefault("uuid", str(uuid4()))
        pages.extend(loaded_pages)
    return pages


def main():
    try:
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        WEAVIATE_URL = os.getenv("WEAVIATE_URL")
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")

        # instantiate embedding model and LLM
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

        # create strategies (pass embeddings for semantic strategies)
        strategies = get_default_strategies(embeddings=embeddings)

        # load PDFs (customize)
        pdf_files = ["Chunking_RAG1.pdf", "final_report_Harshit.pdf"]
        pages = load_pdfs(pdf_files)
        logger.info("Total pages loaded: %d", len(pages))

        # Weaviate client
        from weaviate.auth import AuthApiKey
        import weaviate
        auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)
        client = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=auth_config, headers={"X-OpenAI-Api-Key": OPENAI_API_KEY})
        wm = WeaviateManager(client)

        # embedding cache
        cache = EmbeddingCache()

        # pick a query from user
        query = input("Enter your query: ")

        for name, splitter in strategies.items():
            try:
                logger.info("Running strategy %s", name)
                if hasattr(splitter, 'split_documents'):
                    documents = splitter.split_documents(pages)
                elif callable(splitter):
                    documents = splitter(pages)
                else:
                    raise RuntimeError("Splitter for %s is unusable" % name)

                logger.info("Generated %d chunks", len(documents))

                collection_name = f"Doc_Combined_{name}"
                wm.create_collection(collection_name)

                # ensure each doc has uuid
                for doc in documents:
                    doc.metadata.setdefault("uuid", str(uuid4()))

                vectorstore = WeaviateVectorStore.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    client=client,
                    index_name=collection_name,
                    text_key="text",
                    by_text=False,
                )

                retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
                result = qa_chain.invoke({"query": query})

                print(f"\nStrategy: {name} — Answer:")
                print(result['result'])
                print(f"Chunks used in answer: {len(result['source_documents'])} / {len(documents)}")

            except Exception as e:
                logger.exception("Strategy %s failed: %s", name, e)
                # continue with other strategies

        client.close()
        logger.info("All done")

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        print("Pipeline failed. Check logs for details.")

if __name__ == '__main__':
    main()
