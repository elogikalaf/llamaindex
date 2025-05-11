from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
import os
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
)

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    api_key=GEMINI_API_KEY,
)

Settings.llm = llm

# use custom embed model
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the story about?")
print(response)
