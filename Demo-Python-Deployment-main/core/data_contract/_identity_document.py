from datetime import datetime
from typing import Union, Literal, Text, override

from pydantic import BaseModel, Field, field_validator, ConfigDict

from ._static_enums import NationalityISO3, CountryISO3, Gender, IdentityDocumentType
from utils.helpers import parse_date


class IdentityDocument(BaseModel):
    document_type: Literal[IdentityDocumentType.UNKNOWN] = Field(default=IdentityDocumentType.UNKNOWN, serialization_alias="DocumentType")
    name: Union[Text, None] = Field(default=None, serialization_alias="Name")
    nationality: NationalityISO3 = Field(default=NationalityISO3.UNKNOWN, serialization_alias="Nationality")
    dob: Union[datetime, None] = Field(default=None, serialization_alias="DOB")
    gender: Gender = Field(default=Gender.UNKNOWN, serialization_alias="Gender")
    expiry: Union[datetime, None] = Field(default=None, serialization_alias="ExpiryDate")
    is_expired: bool = Field(default=True, serialization_alias="IsExpired")

    # Document Serialization Configurations
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.date().isoformat()
        }
    )

    # Validators

    @field_validator("gender", mode="before")
    @classmethod
    def _transform_gender(cls, value: Union[Gender, str]) -> "Gender":
        if isinstance(value, str) and len(value) == 1:
            if value.casefold() == "m":
                return Gender.MALE
            elif value.casefold() == "f":
                return Gender.FEMALE
        elif isinstance(value, Gender):
            return value
        return Gender.UNKNOWN
    
    @field_validator("nationality", mode="before")
    @classmethod
    def _transform_nationality(cls, value: Union[NationalityISO3, str]) -> "NationalityISO3":
        if isinstance(value, str):
            return NationalityISO3[value.upper()] if value in NationalityISO3.__members__.keys() else NationalityISO3.UNKNOWN
        elif isinstance(value, NationalityISO3):
            return value
        return NationalityISO3.UNKNOWN
    
    @field_validator("dob", mode="before")
    @classmethod
    def _transform_dob(cls, value: Union[str, datetime]) -> "datetime":
        if isinstance(value, str):
            return parse_date(value)
        return value
    
    @field_validator("expiry", mode="before")
    @classmethod
    def _transform_expiry(cls, value: Union[str, datetime]) -> "datetime":
        if isinstance(value, str):
            return parse_date(value)
        return value
    
    # Overridden methods
    @override
    def model_post_init(self, __conAnyStr):
        if self.expiry is not None:
            self.is_expired = datetime.now() > self.expiry

class Passport(IdentityDocument):
    document_type: Literal[IdentityDocumentType.PASSPORT] = Field(default=IdentityDocumentType.PASSPORT, serialization_alias="DocumentType")
    passport_number: Union[Text, None] = Field(default=None, serialization_alias="PassportNumber")
    passport_check: Union[None, int] = Field(default=None, exclude=True)
    dob_check: Union[int, None] = Field(default=None, exclude=True)
    expiry_check: Union[int, None] = Field(default=None, exclude=True)


    #Validators   
    @field_validator("passport_check", mode="before")
    @classmethod
    def _transform_passport_check(cls, value: Union[str, int]):
        if isinstance(value, str):
            return int(value)
        return value
    
    @field_validator("dob_check", mode="before")
    @classmethod
    def _transform_dob_check(cls, value: Union[str, int]):
        if isinstance(value, str):
            return int(value)
        return value
    
    @field_validator("expiry_check", mode="before")
    @classmethod
    def _transform_expiry_check(cls, value: Union[str, int]):
        if isinstance(value, str):
            return int(value)
        return value
    
class CNIC(IdentityDocument):
    document_type: Literal[IdentityDocumentType.CNIC] = Field(default=IdentityDocumentType.CNIC, serialization_alias="DocumentType")
    cnic_number: Union[Text, None] = Field(default=None, serialization_alias="CNICNumber")
    country_of_stay: CountryISO3 = Field(default=CountryISO3.UNKNOWN, serialization_alias="CountryOfStay")

    @field_validator("country_of_stay", mode="before")
    @classmethod
    def _transform_country_of_stay(cls, value: Union[CountryISO3, str]) -> "CountryISO3":
        if isinstance(value, str):
            return CountryISO3[value.upper()] if value in CountryISO3.__members__.keys() else CountryISO3.UNKNOWN
        elif isinstance(value, CountryISO3):
            return value
        return CountryISO3.UNKNOWN
