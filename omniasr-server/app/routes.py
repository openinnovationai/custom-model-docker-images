"""
API routes for Omnilingual-ASR server.
"""

import logging

from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import MODEL_NAME
from app.exceptions import APIError
from app.handlers import handle_runtime_error
from app.schemas import ModelsResponse, TranscriptionResponse
from app.service import asr_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health-check")
async def health_check():
    """Health check endpoint."""
    return "ok"


@router.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    """Get model information."""
    return {
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "omnilingual-asr",
            }
        ]
    }


@router.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile,
    model: str = Form(default=MODEL_NAME),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: str | None = Form(default=None),
):
    """
    OpenAI Whisper-compatible transcription endpoint.

    Args:
        file: Audio file (wav, mp3, flac, etc.)
        model: Model identifier (informational only)
        language: Language code (ISO 639-1 or Omnilingual-ASR format)
        prompt: Optional prompt (not used)
        response_format: json, verbose_json, text, srt, or vtt
        temperature: Sampling temperature (not used)
        timestamp_granularities: Timestamp detail level (not used)
    """
    if not file.filename:
        logger.warning("Transcription request rejected: no file provided")
        raise APIError(
            status_code=400,
            message="No file provided",
            param="file",
        )

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        logger.warning("Transcription request rejected: empty file")
        raise APIError(
            status_code=400,
            message="Empty file provided",
            param="file",
        )

    logger.info(
        f"Transcription request: file={file.filename}, language={language}, "
        f"format={response_format}"
    )

    try:
        text = await asr_service.transcribe(audio_bytes, language=language)
    except RuntimeError as e:
        logger.exception(f"Transcription failed for {file.filename}")
        handle_runtime_error(e)
    except Exception as e:
        logger.exception(f"Transcription failed for {file.filename}")
        raise APIError(
            status_code=500,
            message=f"Transcription failed: {e}",
            error_type="server_error",
        )

    if response_format == "text":
        return PlainTextResponse(content=text)

    return JSONResponse(content=TranscriptionResponse(text=text).model_dump())
