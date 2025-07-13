from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pinecone
import os

# 🔧 Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# 🔌 Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX)

# 🧠 Load embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 🚀 FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def root():
    return {"message": "Yarey Chatbot API is alive!"}

@app.post("/query")
def query(req: QueryRequest):
    try:
        query_embedding = model.encode(req.question).tolist()
        result = index.query(vector=query_embedding, top_k=req.top_k, include_metadata=True)
        responses = [
            {
                "question": match["metadata"].get("question"),
                "answer_th": match["metadata"].get("answer_th"),
                "answer_en": match["metadata"].get("answer_en"),
                "score": match["score"]
            }
            for match in result["matches"]
        ]
        return {"results": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
