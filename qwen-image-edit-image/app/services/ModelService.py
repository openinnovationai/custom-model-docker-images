from diffusers import QwenImageEditPipeline
import torch
from PIL import Image
import base64
from io import BytesIO
import asyncio
from app.models.EditRequestPayload import EditRequestPayload
from tempfile import NamedTemporaryFile, mkdtemp
import shutil


class ModelService:
    def __init__(self):
        self.__model = None
        self.__model_path = "ovedrive/qwen-image-edit-4bit"
        self.__load_model()

    
    def __load_model(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        if self.__model is None:
            self.__model = QwenImageEditPipeline.from_pretrained(
                self.__model_path, torch_dtype=torch.bfloat16
            )
            self.__model.to("cuda")

    async def edit_image(self, payload: EditRequestPayload) -> str:
        args = {
            "prompt": payload.prompt,
            "negative_prompt": payload.negative_prompt,
            "true_config_scale": payload.true_config_scale,
            "num_inference_steps": payload.num_inference_steps,
        }

        tmp_dir = mkdtemp()
        binary_file = None
        bytes_file = base64.b64decode(payload.image)
        binary_file = BytesIO(bytes_file)

        assert binary_file is not None

        with NamedTemporaryFile(dir=tmp_dir, delete=True) as tmp_file:
            shutil.copyfileobj(binary_file, tmp_file)
            tmp_file_path = tmp_file.name

            with torch.inference_mode():
                try:
                    output = await asyncio.to_thread(lambda: self.__model(image=input_image, generator=torch.manual_seed(0), **args))
                    output_image = output.images[0]
                    buffer = BytesIO()
                    output_image.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    return img_base64
                finally:
                    torch.cuda.empty_cache()

