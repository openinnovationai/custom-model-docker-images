import logging

import pyannote.audio
from data import RTTMSegment, convert_annotation_to_segments
from fastapi import FastAPI, HTTPException, Request
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

    def load_model(self):
        self.model = pyannote.audio.Pipeline(MODELID)

    def diarize(self, **kwargs):
        assert self.model is not None, "Model is not loaded."

        audio_path = kwargs.pop("audio_path")
        annotation = self.model(audio_path)
        return convert_annotation_to_segments(annotation)


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

    @app.post("/diarize")
    async def diarize(self, request: Request):
        body = await request.json()
        output = self.model.diarize(**body)
        return output

    @app.get("/health-check")
    async def get_models(self):
        logger.info("Health check on /models")
        await check_model_liveness()
        return {"model": MODELID}
