from FlagEmbedding import FlagReranker
from ray import serve
from fastapi import FastAPI, Request, HTTPException
import logging

logger = logging.getLogger("ray.serve")
app = FastAPI()

# Define the model ID for bge-reranker-v2-m3
MODELID = "BAAI/bge-reranker-v2-m3"


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
        self.model = FlagReranker(model_name, use_fp16=True, device="cuda")

    def rerank(self, **kwargs) -> dict:
        """Rerank sentences based on their relevance."""
        assert self.model is not None, "Model is not loaded."

        texts = kwargs.pop("texts")
        query = kwargs.pop("query")

        pairs = [[query, candidate] for candidate in texts]
        output = self.model.compute_score(pairs, **kwargs)
        return {"scores": output}


@serve.deployment()
@serve.ingress(app)
class App:
    def __init__(self):
        """Initialize the FastAPI app with the loaded model."""
        self.model = None
        self.__load_model()

    def __load_model(self):
        """Ensure the model is loaded only once."""
        assert self.model is None, "Model is already loaded."
        model = Model()
        model.load_model()
        self.model = model

    @app.post("/rerank")
    async def rerank(self, request: Request):
        """Endpoint to rerank sentences based on the model."""
        body = await request.json()

        # Log the input request
        logger.info(f"Inference params: {body}")

        # Call the model's reranking method and return the result
        output = self.model.rerank(**body)
        return output

    @app.get("/health-check")
    async def get_models(self):
        """Endpoint to check the health of the service."""
        logger.info("Health check on /models")
        await check_model_liveness()
        return {"model": MODELID}
