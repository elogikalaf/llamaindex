from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from dotenv import load_dotenv
import os
import sys

# Load environment variables from .env file
load_dotenv()


DATA_URL = "https://virgool.io/Mobarak/%D8%AF%D8%A7%D8%B3%D8%AA%D8%A7%D9%86-%DA%A9%D9%88%D8%AA%D8%A7%D9%87-%DA%A9%D9%88%D8%AF%D8%AA%D8%A7%DB%8C-%D8%A8%D9%87%D8%A7%D8%B1%DB%B1-r5ucftjluvqu"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API KEY NOT FOUND")


embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    api_key=GEMINI_API_KEY,
)

llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
)

Settings.llm = llm

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pinecone_client.Index("hanji")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

query_engine = RetrieverQueryEngine(retriever=retriever)

print("Enter your query (Press Ctrl+D to finish on Linux/macOS or Ctrl+Z + Enter on Windows):")
query = sys.stdin.read().strip()
response = query_engine.query(query)

print()
print(response)
