"""
Exception handlers for OpenAI-compatible error responses.
"""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import APIError
from app.schemas import ErrorResponse

logger = logging.getLogger(__name__)


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions with OpenAI-compatible error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorResponse.ErrorInfo(
                message=exc.message,
                type=exc.error_type,
                param=exc.param,
                code=exc.code,
            )
        ).model_dump(),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors with OpenAI-compatible error responses."""
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        loc = first_error.get("loc", ())
        param = ".".join(str(x) for x in loc) if loc else None
        message = first_error.get("msg", "Validation error")
    else:
        param = None
        message = "Validation error"

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorResponse.ErrorInfo(
                message=message,
                type="invalid_request_error",
                param=param,
            )
        ).model_dump(),
    )


def handle_runtime_error(e: RuntimeError) -> None:
    """Handle runtime errors with OpenAI-compatible error responses."""
    cause = e.__cause__
    cause_msg = str(cause).lower() if cause else ""

    if "sndfile" in cause_msg or "decode" in cause_msg:
        raise APIError(
            status_code=400,
            message="Could not decode audio file. The file may be corrupted or in an unsupported format.",
            param="file",
            code="invalid_audio_format",
        )

    if "max audio length" in cause_msg:
        raise APIError(
            status_code=400,
            message="Audio file is too long. The maximum audio length is 40 seconds.",
            param="file",
            code="invalid_audio_length",
        )

    raise APIError(
        status_code=500,
        message=f"Transcription failed: {e}. Cause: {cause_msg or 'unknown'}",
        error_type="server_error",
    )
