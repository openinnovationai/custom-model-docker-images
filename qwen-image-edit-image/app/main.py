from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.controllers.ModelController import ModelController
from app.models.EditRequestPayload import EditRequestPayload
from app.models.EditResponse import EditResponse
import logging
import time

model_controller = ModelController()

app = FastAPI(title="Qwen Image Edit API", version="0.1.0")


logger = logging.getLogger("qwen_image_edit")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger.setLevel(logging.INFO)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Starting Qwen Image Edit API")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Shutting down Qwen Image Edit API")


@app.get("/health-check")
async def health() -> JSONResponse:
    logger.info("Health check requested")
    return JSONResponse({"status": "ok"})


@app.post("/v1/images/edits")
async def edit_image(payload: EditRequestPayload) -> EditResponse:
    start_time = time.perf_counter()
    logger.info(
        "Received image edit request: prompt=%r, negative_prompt=%s, true_config_scale=%.2f, num_inference_steps=%d, image_len=%d",
        payload.prompt[:80] if payload.prompt else "",
        "present" if payload.negative_prompt else "none",
        payload.true_config_scale,
        payload.num_inference_steps,
        len(payload.image) if payload.image else 0,
    )
    try:
        response = await model_controller.edit_image(payload)
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Image edit completed in %.1f ms", duration_ms)
        return response
    except Exception:
        logger.exception("Image edit failed")
        raise
