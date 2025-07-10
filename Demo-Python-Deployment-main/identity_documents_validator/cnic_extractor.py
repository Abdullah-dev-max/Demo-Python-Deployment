import json
import os
import re
from typing import Text, override

import requests
from dotenv import load_dotenv

from core.data_contract import CNIC, IdentityDocumentType
from .id_document_extractor_base import BaseIDExtractor

load_dotenv()


class CNICExtractor(BaseIDExtractor):
    _URL = r"https://api2-eu.idanalyzer.com"
    _API_KEY = os.environ.get("ID_ANALYZER_API_KEY")
    
    def __init__(self):
        super().__init__()

    @override
    @classmethod
    def get_text_from_image(cls, b64_image_string: Text) -> "CNIC":
        def remove_word_name(text: Text) -> Text:
            name_matches = re.search(r"Name", text)
            if name_matches is not None:
                if name_matches.end() == 4:
                    if len(text) > 4 and text[4].isspace():
                        text = text[name_matches.end():].strip()
            return text.strip()

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-KEY": cls._API_KEY
        }
        payload = {
            "profile": "security_none",
            "document": b64_image_string
        }

        response = requests.post(url=f"{cls._URL}/scan", headers=headers, json=payload)
        data = json.loads(response.text)
        if response.status_code == 200:
            doc_meta_data = CNIC()
            if data.get("success", False) is True:
                meta_data = {}
                dob = data.get("data", {}).get("dob", [])
                if dob is not None and len(dob) > 0:
                    meta_data["dob"] = dob.pop(0).get("value")
                expiry = data.get("data", {}).get("expiry", [])
                if expiry is not None and len(expiry) > 0:
                    meta_data["expiry"] = expiry.pop(0).get("value")
                nationality = data.get("data", {}).get("nationalityIso3", [])
                if nationality is not None and len(nationality) > 0:
                    meta_data["nationality"] = nationality.pop(0).get("value")
                cnic_number = data.get("data", {}).get("documentNumber", [])
                if cnic_number is not None and len(cnic_number) > 0:
                    meta_data["cnic_number"] = cnic_number.pop(0).get("value")
                first_name = data.get("data", {}).get("firstName", [])
                last_name = data.get("data", {}).get("lastName", [])
                full_name = list()
                if first_name is not None and len(first_name) > 0:
                    full_name.append(first_name.pop(0).get("value"))
                if last_name is not None and len(last_name) > 0:
                    full_name.append(last_name.pop(0).get("value"))
                meta_data["name"] = " ".join(full_name).strip().title() if len(full_name) > 0 else None
                meta_data["name"] = remove_word_name(meta_data["name"]).title() if meta_data["name"] is not None else None
                gender = data.get("data", {}).get("sex", [])
                if gender is not None and len(gender) > 0:
                    meta_data["gender"] = gender.pop(0).get("value")
                country_of_stay = data.get("data", {}).get("countryIso3", [])
                if country_of_stay is not None and len(country_of_stay) > 0:
                    meta_data["country_of_stay"] = country_of_stay.pop(0).get("value")
                doc_meta_data: CNIC = CNIC.model_validate(meta_data)
                doc_meta_data.document_type = IdentityDocumentType.CNIC
            return doc_meta_data
        else:
            error = data["error"]["code"]
            message = data["error"]["message"]
            status_code = data["error"]["status"]
            raise Exception(f"Error: {error}. Message: {message} Status code: {status_code}")
        
        

    