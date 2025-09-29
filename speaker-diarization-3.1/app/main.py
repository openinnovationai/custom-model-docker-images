from fastapi import FastAPI, HTTPException, Request, UploadFile
from .model import Model
import os
import logging
import tempfile
import aiofiles
import asyncio


logger = logging.getLogger("speaker_diarization.main")

model = Model()

app = FastAPI()

@app.on_event("startup")
def startup_event():
    logger.info("Loading model 🔄")
    model.load_model()
    logger.info("Model loaded ✅")


@app.get("/health-check")
async def health_check():
    if model.pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}

@app.post("/diarize")
async def diarize(request: Request):
    assert model.pipeline is not None, "Model is not loaded."
    # Accept one of: JSON {"url": "https://example.com/audio.wav"}, multipart form (file field), or raw bytes
    content_type = request.headers.get("content-type", "").lower()

    temp_path = None
    try:
        if content_type.startswith("application/json"):
            body = await request.json()
            audio_url = body.get("url")
            if audio_url:
                temp_path = model.download_url_to_temp(audio_url)
                audio_path = temp_path
            else:
                raise HTTPException(
                    status_code=400, detail="Missing 'audio_path' or 'url' in JSON body.")
        elif content_type.startswith("multipart/form-data"):
            form = await request.form()
            file: UploadFile = form.get("file")
            if file is None:
                raise HTTPException(
                    status_code=400, detail="Missing 'file' field in multipart form.")
            suffix = os.path.splitext(file.filename or "")[1] or ".wav"
            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(await file.read())
            finally:
                await file.close()
            audio_path = temp_path
        else:
            # Treat as raw bytes
            data = await request.body()
            if not data:
                raise HTTPException(
                    status_code=400, detail="Empty request body.")
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            async with aiofiles.open(temp_path, "wb") as out:
                await out.write(data)
            audio_path = temp_path

        # Offload CPU-bound diarization to a worker thread to avoid blocking
        output = await asyncio.to_thread(model.diarize, audio_path=audio_path)
        return output
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                # Best-effort cleanup
                pass