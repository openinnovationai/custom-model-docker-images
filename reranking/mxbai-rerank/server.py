import os
import logging
import torch
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import uvicorn

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("reranker-server")

MODEL_PATH = os.getenv("OICM_MODEL_PATH")
model: Optional[CrossEncoder] = None

def get_device():
    """Detects the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    device = get_device()
    logger.info(f"Starting server... Device selected: {device}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model path not found: {MODEL_PATH}")
    
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = CrossEncoder(
            MODEL_PATH, 
            device=device,
            local_files_only=True
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise e
    
    yield
    
    logger.info("Shutting down server...")
    model = None

class RankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None

class RankResult(BaseModel):
    corpus_id: int
    score: float
    text: str

app = FastAPI(title="Reranker API", lifespan=lifespan)

@app.get("/health-check")
def health_check():
    """Simple health check for load balancers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(model.device)}

@app.post("/v1/rerank", response_model=List[RankResult])
def rank_documents(request: RankRequest):
    """
    Ranks documents based on the query.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    try:
        results = model.rank(
            request.query, 
            request.documents, 
            return_documents=True, 
            top_k=request.top_k
        )
        return results

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)