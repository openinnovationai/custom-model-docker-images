"""Async ASR service for Omnilingual-ASR model."""

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from app.config import MODEL_NAME
from app.languages import map_whisper_to_omnilingual

logger = logging.getLogger(__name__)


class OmnilingualASRService:
    """Async ASR service wrapping the Omnilingual-ASR pipeline."""

    def __init__(self):
        self.pipeline: ASRInferencePipeline | None = None

    def load_model(self) -> None:
        """Load the ASR model. Called once at startup."""

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        # Use float16 for compute capability < 8.0 (e.g. T4)
        dtype = torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability() < (8, 0):
            dtype = torch.float16

        logger.info(f"Loading model {MODEL_NAME} on {device}...")
        self.pipeline = ASRInferencePipeline(
            model_card=MODEL_NAME, device=device, dtype=dtype
        )
        logger.info(f"Model {MODEL_NAME} loaded successfully on {device}")

    @property
    def is_llm_model(self) -> bool:
        """Check if the current model is an LLM-based model (supports language conditioning)."""

        return "LLM" in MODEL_NAME

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes
            language: Optional language code (OpenAI or Omnilingual-ASR format)

        Returns:
            Transcribed text
        """

        if self.pipeline is None:
            logger.error("Transcription attempted before model was loaded")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Map language code if provided and model supports it
        lang_param = None
        if language and self.is_llm_model:
            lang_param = map_whisper_to_omnilingual(language)
            logger.debug(f"Language mapped: {language} -> {lang_param}")

        # Run transcription (sync, but wrapped for async compatibility)
        audio_size_kb = len(audio_bytes) / 1024
        logger.info(
            f"Starting transcription: {audio_size_kb:.1f}KB, language={lang_param or 'auto'}"
        )

        if lang_param:
            transcriptions = self.pipeline.transcribe(
                [audio_bytes], lang=[lang_param], batch_size=1
            )
        else:
            transcriptions = self.pipeline.transcribe([audio_bytes], batch_size=1)

        result = transcriptions[0] if transcriptions else ""
        logger.info(f"Transcription complete: {len(result)} chars")
        return result


# Global service instance
asr_service = OmnilingualASRService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler - load model on startup."""

    asr_service.load_model()
    yield
