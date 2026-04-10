from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    depth: int = 3
    queries: int = 3
    fmt: str = "report"

@app.post("/search")
async def search(body: SearchRequest):
    result = await run(body.query)
    return {"report": result, "success": True}

@app.get("/health")
async def health():
    return {"status": "ok"}