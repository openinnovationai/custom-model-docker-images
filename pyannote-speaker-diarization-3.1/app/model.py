import asyncio
import logging
import os
import tempfile
from typing import List, Optional
from urllib.parse import urlparse

import aiofiles
import httpx
import pyannote.audio
import torch
from data import RTTMSegment, convert_annotation_to_segments
from fastapi import FastAPI, HTTPException, Request, UploadFile
from ray import serve

logger = logging.getLogger("ray.serve")
app = FastAPI()

MODELID = "pyannote/speaker-diarization-3.1"


async def check_model_liveness():
    status = serve.status()
    is_running = status.applications["model"].status == "RUNNING"
    if not is_running:
        raise HTTPException(
            status_code=503, detail="Model is not currently available.")
    return True


class Model:
    def __init__(self):
        self.model = None

    def load_model(self) -> None:
        self.model = pyannote.audio.Pipeline.from_pretrained(MODELID)
        # Move pipeline to best available device (CUDA > MPS > CPU)
        # Log CUDA devices
        if torch.cuda.is_available():
            try:
                num = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(num)]
                logger.info(f"CUDA available: {num} device(s): {names}")
            except Exception as e:
                logger.warning(f"Failed querying CUDA devices: {e}")
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) backend is available")
            device = torch.device("mps")
        else:
            logger.info("Falling back to CPU (no CUDA/MPS detected)")
            device = torch.device("cpu")
        try:
            self.model.to(device)
            logger.info(f"Pipeline moved to device: {device}")
        except Exception as e:
            logger.warning(
                f"Failed to move pipeline to {device}: {e}. Falling back to CPU.")
            try:
                self.model.to(torch.device("cpu"))
            except Exception:
                pass
        logger.info(f"Model loaded: {MODELID}")

    def diarize(self, **kwargs) -> List[RTTMSegment]:
        assert self.model is not None, "Model is not loaded."

        audio_path = kwargs.pop("audio_path")
        annotation = self.model(audio_path)
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        return convert_annotation_to_segments(annotation, file_id)


@serve.deployment()
@serve.ingress(app)
class App:
    def __init__(self):
        self.model = None
        self.__load_model()

    def __load_model(self):
        assert self.model is None
        self.model = Model()
        self.model.load_model()

    async def _download_url_to_temp(self, url: str) -> str:
        """Download a URL to a temporary file asynchronously and return the path.

        Uses streaming to avoid large memory usage and preserves file suffix when possible.
        """
        suffix = os.path.splitext(urlparse(url).path)[1] or ".wav"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    async with aiofiles.open(temp_path, "wb") as out_f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            if chunk:
                                await out_f.write(chunk)
        except httpx.HTTPError as e:
            # Cleanup partial file on failure
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            raise HTTPException(
                status_code=400, detail=f"Failed to download URL: {e}")
        return temp_path

    @app.post("/diarize")
    async def diarize(self, request: Request):
        # Accept one of: JSON {"audio_path": "/path"}, multipart form (file field), or raw bytes
        content_type = request.headers.get("content-type", "").lower()

        temp_path: Optional[str] = None
        try:
            if content_type.startswith("application/json"):
                body = await request.json()
                audio_path: Optional[str] = body.get("audio_path")
                audio_url: Optional[str] = body.get("url")
                if not audio_path and audio_url:
                    temp_path = await self._download_url_to_temp(audio_url)
                    audio_path = temp_path
                if not audio_path:
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
            output = await asyncio.to_thread(self.model.diarize, audio_path=audio_path)
            return output
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    # Best-effort cleanup
                    pass

    @app.get("/health-check")
    async def get_models(self) -> dict:
        logger.info("Health check on /models")
        await check_model_liveness()
        return {"model": MODELID}
