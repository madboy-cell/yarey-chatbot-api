from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("pcsk_fsCk9_JXNZU6pfDhgtJdwcXxmhxNrtWzcCCcwjw755ZAEqhMp1MTQnsorD8AxfJ5fuMkG"))
index = pc.Index("yarey-chatbot")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("message")
    vector = model.encode(query).tolist()
    results = index.query(vector=vector, top_k=1, include_metadata=True)
    match = results['matches'][0]['metadata']
    return {
        "reply_th": match.get("answer_th", ""),
        "reply_en": match.get("answer_en", "")
    }
