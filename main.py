from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
import pinecone

app = FastAPI()

# Pinecone setup
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

class Query(BaseModel):
    question: str

@app.post("/query")
def query_bot(q: Query):
    embedding = model.encode(q.question).tolist()
    response = index.query(vector=embedding, top_k=3, include_metadata=True)
    results = [match["metadata"] for match in response["matches"]]
    return {"answer": results}
