from transformers import pipeline
from ray import serve
from fastapi import FastAPI, Request, HTTPException
import logging
import aiohttp
import tempfile
import os

logger = logging.getLogger("ray.serve")
app = FastAPI()

MODELID = "MelodyMachine/Deepfake-audio-detection-V2"


async def check_model_liveness():
    """Check if the model service is running."""
    status = serve.status()
    is_running = status.applications["model"].status == "RUNNING"
    if not is_running:
        raise HTTPException(status_code=503, detail="Model is not currently available.")
    return True


class Model:
    def __init__(self) -> None:
        """Initialize the model object."""
        self.model = None

    def load_model(self, model_name: str = MODELID) -> None:
        """Load the BGE ReRanker model using FlagReranker."""
        self.classifier = pipeline("audio-classification", model=model_name)

    def classify(self, audio_path: str) -> dict:
        """Classify the audio file."""
        return self.classifier(audio_path)


@serve.deployment()
@serve.ingress(app)
class App:
    def __init__(self):
        """Initialize the FastAPI app with the loaded model."""
        print("Initializing model...")
        self.model = None
        self.__load_model()

    def __load_model(self):
        """Ensure the model is loaded only once."""
        assert self.model is None, "Model is already loaded."
        model = Model()
        model.load_model()
        self.model = model

    @app.post("/classify")
    async def classify(self, request: Request):
        """Endpoint to classify the audio file."""
        body = await request.json()

        audio_url = body.get("audio_url")
        if not audio_url:
            raise HTTPException(status_code=400, detail="audio_url is required")

        # Log the input request
        logger.info(f"Inference params: {body}")

        # Download the audio file to a temp file
        temp_file_path = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download audio file: {resp.status}")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_file_path = tmp.name
                        while True:
                            chunk = await resp.content.read(1024)
                            if not chunk:
                                break
                            tmp.write(chunk)
            # Call the model's classify method with the temp file path
            output = self.model.classify(temp_file_path)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        return output

    @app.get("/health-check")
    async def get_models(self):
        """Endpoint to check the health of the service."""
        logger.info("Health check on /models")
        await check_model_liveness()
        return {"model": MODELID}
