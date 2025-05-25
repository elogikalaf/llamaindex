from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
import os

# Load environment variables from .env file
load_dotenv()
DATA_URL=""
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API KEY NOT FOUND")

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    api_key=GEMINI_API_KEY,
)

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


pinecone_index = pinecone_client.Index("hanji")


# Check for existing records and delete if any
index_stats = pinecone_index.describe_index_stats()
if index_stats.get("total_vector_count", 0) > 0:
    print("Existing vectors found. Deleting all...")
    pinecone_index.delete(delete_all=True)
else:
    print("Index is empty. Proceeding with ingestion.")


vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)

# web scarping
# loader = BeautifulSoupWebReader()
# documents = loader.load_data(urls=[DATA_URL])

# dir reading
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        embed_model,
    ],
    vector_store=vector_store,
)

pipeline.run(documents=documents)
