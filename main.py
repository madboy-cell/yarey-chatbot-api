from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Pinecone and embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_pinecone(request: QueryRequest):
    embedding = model.encode(request.query).tolist()
    result = index.query(vector=embedding, top_k=3, include_metadata=True)
    return result
