from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_groq import ChatGroq

app = FastAPI()

llm = ChatGroq(
    api_key="",
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)
async def stream_llm(prompt: str):
    for chunk in llm.stream(prompt):
        yield chunk.content or ""

@app.get("/chat")
async def chat(q: str):
    return StreamingResponse(stream_llm(q), media_type="text/plain")