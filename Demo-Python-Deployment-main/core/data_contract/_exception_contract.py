from typing import Text
from pydantic import BaseModel, Field


class CustomException(BaseModel):
    error: str = Field(serialization_alias="Error")
    status_code: int = Field(serialization_alias="StatusCode")
    message: Text = Field(serialization_alias="Message")
    details: Text = Field(serialization_alias="Details")

