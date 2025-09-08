from pydantic import BaseModel, Field
import datetime


class EditResponse(BaseModel):
    class Data(BaseModel):
        b64_json: str

    created: datetime.datetime
    usage: dict = Field(default_factory=dict)
    data: list[Data]
    