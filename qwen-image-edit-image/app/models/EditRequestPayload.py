from pydantic import BaseModel


class EditRequestPayload(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    true_config_scale: float = 4.0
    num_inference_steps: int = 10
    image: str