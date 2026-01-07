"""
Entry point for the Omnilingual-ASR FastAPI server.
"""

import logging
import os

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    port = int(os.environ.get("OMNILINGUAL_PORT", "8080"))
    host = os.environ.get("OMNILINGUAL_HOST", "0.0.0.0")

    logger.info(f"Starting Omnilingual-ASR server on {host}:{port}")

    uvicorn.run("app.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
