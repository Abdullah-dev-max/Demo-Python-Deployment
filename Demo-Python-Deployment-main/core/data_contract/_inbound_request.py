from typing import Union, Text
from pydantic import BaseModel, Field

from ._static_enums import IdentityDocumentType


class FaceRegisterInboundRequest(BaseModel):
    b64_scanned_img: Text = Field(alias="Base64ScannedImage")
    user_id: Union[Text, None] = Field(default=None, alias="UserId")


class OCRInboundRequest(BaseModel):
    b64_scanned_img: Text = Field(alias="Base64ScannedImage")
    point_id: Union[Text, None] = Field(default=None, alias="PointId")
    user_id: Union[Text, None] = Field(default=None, alias="UserId")
    document_type: IdentityDocumentType = Field(default=IdentityDocumentType.UNKNOWN, alias="DocumentType")


class FaceDeleteInboundRequest(BaseModel):
    point_id: Text = Field(default=None, alias="PointId")
    user_id: Text = Field(default=None, alias="UserId")


class FaceUpdateInboundRequest(FaceRegisterInboundRequest):
    point_id: Text = Field(alias="PointId")
