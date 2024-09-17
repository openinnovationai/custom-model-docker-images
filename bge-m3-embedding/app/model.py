from FlagEmbedding import BGEM3FlagModel

from ray import serve
from fastapi import FastAPI, Request, HTTPException
import logging


logger = logging.getLogger("ray.serve")
app = FastAPI()

MODELID = "BAAI/bge-m3"

async def check_model_liveness():
    status = serve.status()
    is_running = status.applications["model"].status == "RUNNING"
    if not is_running:
        raise HTTPException(status_code=503, detail="Model is not currently available.")
    return True


class Model:
    def __init__(self) -> None:
        self.model = None

    def load_model(self, model_name: str = MODELID) -> None:
        self.model = BGEM3FlagModel(model_name, use_fp16=True, device="cuda")

    async def generate(self, **kwargs) -> dict:
        assert self.model is not None

        sentences = kwargs.pop("sentences")
        output = self.model.encode(sentences, **kwargs)
        output["dense_vecs"] = output["dense_vecs"].tolist()
        return output


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

    @app.post("/generate")
    async def generate(self, request: Request):
        body = await request.json()

        # TODO: validate the request body

        logger.info(f"Inference params: {body}")
        output = await self.model.generate(**body)
        return output

    @app.get("/health-check")
    async def get_models(self):
        logger.info("calling /models")
        await check_model_liveness()
        return {"model": MODELID}
