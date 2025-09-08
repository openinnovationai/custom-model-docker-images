from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.controllers.ModelController import ModelController
from app.models.EditRequestPayload import EditRequestPayload
from app.models.EditResponse import EditResponse

model_controller = ModelController()

app = FastAPI(title="Qwen Image Edit API", version="0.1.0")


@app.get("/health-check")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/v1/images/generations")
async def edit_image(payload: EditRequestPayload) -> EditResponse:
    return model_controller.edit_image(payload)
