from app.services.ModelService import ModelService
from fastapi import UploadFile
from app.models.EditResponse import EditResponse
import datetime



class ModelController:
    def __init__(self, model_service: ModelService | None = None):
        self.__model_service = model_service or ModelService()

    async def edit_image(self, file: UploadFile) -> EditResponse:
        edit_response = await self.__model_service.edit_image(file)
        return EditResponse(
            created=datetime.datetime.now(),
            usage={},
            data=[EditResponse.Data(b64_json=edit_response)]
        )
