from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run

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
    # depth, queries, and fmt are currently placeholders for future expansion
    # as the underlying main.run logic is fixed to the agent pipeline
    result = await run(body.query)

    return {"report": result, "success": True}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)