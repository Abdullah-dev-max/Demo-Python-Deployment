from typing import Union, Text, Annotated
from pydantic import BaseModel, Field

from ._static_enums import APIStatus, IdentityDocumentType
from ._identity_document import IdentityDocument, Passport, CNIC


class FaceOutboundResponse(BaseModel):
    status: APIStatus = Field(default=APIStatus.UNKNOWN, serialization_alias="Status")
    point_id: Union[Text, None] = Field(default=None, serialization_alias="PointId")
    user_id: Union[Text, None] = Field(default=None, serialization_alias="UserId")
    message: Text = Field(default="", serialization_alias="Message")


class OCROutboundResponse(BaseModel):
    status: APIStatus = Field(default=APIStatus.UNKNOWN, serialization_alias="Status")
    point_id: Union[Text, None] = Field(default=None, serialization_alias="PointId")
    user_id: Union[Text, None] = Field(default=None, serialization_alias="UserId")
    message: Text = Field(default="", serialization_alias="Message")
    data: Union[
        Annotated[IdentityDocument, IdentityDocumentType.UNKNOWN],
        Annotated[Passport, IdentityDocumentType.PASSPORT],
        Annotated[CNIC, IdentityDocumentType.CNIC]
    ] = Field(default=IdentityDocument(), serialization_alias="Data")
