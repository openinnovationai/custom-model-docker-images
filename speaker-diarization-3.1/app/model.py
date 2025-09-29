from pyannote.audio import Pipeline
import torch
from .data import RTTMSegment, convert_annotation_to_segments
import os
from typing import List
from urllib.parse import urlparse
import tempfile
import httpx
import aiofiles

import logging

logger = logging.getLogger("speaker_diarization.model")

MODEL_ID = "pyannote/speaker-diarization-3.1"


class Model:
    def __init__(self):
        self.pipeline = None

    def load_model(self):
        logger.info("Loading model %s", MODEL_ID)
        self.pipeline = Pipeline.from_pretrained(MODEL_ID)

        logger.info("Model loaded")
        if torch.cuda.is_available():
            logger.info("Moving model to GPU")
            self.pipeline.to(torch.device("cuda"))
            logger.info("Model moved to GPU")
        else:
            logger.info("CUDA is not available. Model will run on CPU.")


    def download_url_to_temp(self, url: str) -> str:
        """Download a URL to a temporary file asynchronously and return the path.

        Uses streaming to avoid large memory usage and preserves file suffix when possible.
        """
        suffix = os.path.splitext(urlparse(url).path)[1] or ".wav"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with aiofiles.open(temp_path, "wb") as out_f:
                        for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            if chunk:
                                out_f.write(chunk)
        except httpx.HTTPError as e:
            # Cleanup partial file on failure
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                raise e
           
        return temp_path

    
    def diarize(self, **kwargs) -> List[RTTMSegment]:
        assert self.pipeline is not None, "Model is not loaded."

        audio_path = kwargs.pop("audio_path")
        annotation = self.pipeline(audio_path)
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        return convert_annotation_to_segments(annotation, file_id)
