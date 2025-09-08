from diffusers import QwenImageEditPipeline
import torch
from PIL import Image
import base64
from io import BytesIO
import asyncio
from app.models.EditRequestPayload import EditRequestPayload
from tempfile import NamedTemporaryFile, mkdtemp
import shutil
import logging
import time


class ModelService:
    def __init__(self):
        self.__model = None
        self.__model_path = "ovedrive/qwen-image-edit-4bit"
        self.__logger = logging.getLogger("qwen_image_edit.model_service")
        self.__logger.info("Initializing ModelService with model_path=%s", self.__model_path)
        self.__load_model()

    
    def __load_model(self):
        if not torch.cuda.is_available():
            self.__logger.error("CUDA is not available; cannot load model")
            raise ValueError("CUDA is not available")

        if self.__model is None:
            self.__logger.info("Loading QwenImageEditPipeline from %s", self.__model_path)
            start_time = time.perf_counter()
            self.__model = QwenImageEditPipeline.from_pretrained(
                self.__model_path, torch_dtype=torch.bfloat16
            )
            self.__model.to("cuda")
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.__logger.info("Model loaded and moved to CUDA in %.1f ms", duration_ms)

    async def edit_image(self, payload: EditRequestPayload) -> str:
        args = {
            "prompt": payload.prompt,
            "negative_prompt": payload.negative_prompt,
            "num_inference_steps": payload.num_inference_steps,
        }

        start_time = time.perf_counter()
        self.__logger.info(
            "Edit request: prompt=%r, negative_prompt=%s, num_inference_steps=%d, image_len=%d",
            payload.prompt[:80] if payload.prompt else "",
            "present" if payload.negative_prompt else "none",
            payload.num_inference_steps,
            len(payload.image) if payload.image else 0,
        )

        tmp_dir = mkdtemp()
        binary_file = None
        bytes_file = base64.b64decode(payload.image)
        self.__logger.info("Decoded input image bytes: %d", len(bytes_file))
        binary_file = BytesIO(bytes_file)

        assert binary_file is not None

        with NamedTemporaryFile(dir=tmp_dir, delete=True) as tmp_file:
            shutil.copyfileobj(binary_file, tmp_file)
            tmp_file_path = tmp_file.name
            self.__logger.info("Wrote temporary image file: %s", tmp_file_path)

            with torch.inference_mode():
                try:
                    self.__logger.info("Invoking image edit pipeline")
                    output = await asyncio.to_thread(lambda: self.__model(image=tmp_file_path, generator=torch.manual_seed(0), **args))
                    output_image = output.images[0]
                    buffer = BytesIO()
                    output_image.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.__logger.info("Image edit succeeded in %.1f ms", duration_ms)
                    return img_base64
                except Exception:
                    self.__logger.exception("Image edit failed")
                    raise
                finally:
                    torch.cuda.empty_cache()

