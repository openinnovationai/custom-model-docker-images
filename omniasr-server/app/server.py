"""
FastAPI server with OpenAI Whisper-compatible API for Omnilingual-ASR.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app import __version__
from app.exceptions import APIError
from app.handlers import api_error_handler, validation_error_handler
from app.routes import router
from app.service import lifespan

app = FastAPI(
    title="Omnilingual-ASR Server",
    description="OpenAI Whisper-compatible API for Omnilingual-ASR",
    version=__version__,
    lifespan=lifespan,
)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.include_router(router)
