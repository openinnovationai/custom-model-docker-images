import whisper
import torch
import logging
import os
import shutil
import tempfile
import asyncio
from functools import partial
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VERBOSE_LOGGING = os.getenv("OICM_VERBOSE_LOGGING", False) == "True"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    MODEL_NAME = "turbo"
    logger.info(f"Loading model '{MODEL_NAME}' on {device}...")
    ml_models["whisper"] = whisper.load_model(MODEL_NAME, device=device)
    logger.info("Model loaded successfully")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health-check")
async def health_check():
    return {"status": "ok"}

@app.post("/asr")
async def asr(audio: UploadFile = File(...), language: str | None = Form(None)):
    logger.info(f"Received request - language: {language}")
    
    # Whisper requires a file path (str) or numpy array, not a file-like object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name
    audio.file.close()

    logger.info(f"Starting transcription on {tmp_path}...")
    
    loop = asyncio.get_running_loop()
    try:
        model = ml_models["whisper"]
        # Use partial to pass arguments to the synchronous function
        result = await loop.run_in_executor(
            None, 
            partial(model.transcribe, tmp_path, language=language, verbose=VERBOSE_LOGGING)
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    logger.info("Transcription completed")
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
