import base64
import io
import re
from threading import Lock
from typing import Dict, Text, Union, override

import numpy as np
import torch
import paddle
from PIL import Image
from paddleocr import PaddleOCR

from core.data_contract import Passport, IdentityDocumentType
from .id_document_extractor_base import BaseIDExtractor

MODEL_LOCK = Lock()


class PassportExtractor(BaseIDExtractor):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _initiate_paddle_ocr():
        is_paddle_paddle_gpu = paddle.device.is_compiled_with_cuda()
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm='CRNN', max_text_length=44, use_gpu=is_paddle_paddle_gpu, show_log=False)
        return _ocr
        
    _OCR_ENGINE: PaddleOCR = _initiate_paddle_ocr()

    @override
    @classmethod
    def get_text_from_image(cls, b64_image_string: str) -> "Passport":
        image_data = base64.b64decode(b64_image_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        with MODEL_LOCK:
            results = cls._OCR_ENGINE.ocr(image_np, cls=True)
        result = results[0]
        doc_meta_data = Passport()
        if result is not None and len(result) > 0:
            txts = [cls._clean_text(line[1][0]) for line in result]
            txts = list(filter(lambda x: len(x) != 0, txts))
            MRZ = txts[-2:]
            if not MRZ[0].isalnum():
                MRZ[0] = txts[-3]

            mrz_first_line_data = cls._parse_first_mrz_line(mrz=MRZ[0])
            mrz_second_line_data = cls._parse_seconf_mrz_line(mrz=MRZ[1])
            if mrz_second_line_data:
                doc_meta_data: Passport = Passport.model_validate(mrz_second_line_data)
                if mrz_first_line_data:
                    doc_meta_data.name = mrz_first_line_data.get("full_name")
                doc_meta_data.document_type = IdentityDocumentType.PASSPORT
        return doc_meta_data



    
    @staticmethod
    def _clean_text(text: Text):
        return re.sub(r'[^A-Za-z0-9\s.-]', '', text)  # Keeps only letters, numbers, and spaces

    @staticmethod
    def _parse_first_mrz_line(mrz: Text) -> Union[Dict, None]:
        patterns_mrz_line_1 = {
            "document_type":        r"(?P<document_type>[A-Z])",
            "issuing_authority":    r"(?P<issuing_authority>[A-Z]{3})",
            "full_name":            r"(?P<full_name>[A-Z]*)",  # Names part without `<`
        }
        full_pattern_mrz_line_1 = "^"
        for key in patterns_mrz_line_1:
            full_pattern_mrz_line_1 += f"({patterns_mrz_line_1[key]})"
        
        regex_mrz_line_1 = re.compile(full_pattern_mrz_line_1)
        mrz_line_1_results = regex_mrz_line_1.match(mrz.upper())
        if mrz_line_1_results:
            return mrz_line_1_results.groupdict()
        return None
    
    @staticmethod
    def _parse_seconf_mrz_line(mrz: Text) -> Union[Dict, None]:
        patterns_mrz_line_2 = {
            "passport_number": r"(?P<passport_number>[A-Z]{1,2}[0-9]{7})",
            "passport_check":  r"(?P<passport_check>\d)",
            "nationality":     r"(?P<nationality>[A-Z]{3})",
            "dob":             r"(?P<dob>\d{6})",
            "dob_check":       r"(?P<dob_check>\d)",
            "gender":          r"(?P<gender>[MF])",
            "expiry":          r"(?P<expiry>\d{6})",
            "expiry_check":    r"(?P<expiry_check>\d)"
        }
        full_pattern_mrz_line_2 = "^"
        for key in patterns_mrz_line_2:
            full_pattern_mrz_line_2 += f"({patterns_mrz_line_2[key]})?"
        regex_mrz_line_2 = re.compile(full_pattern_mrz_line_2)
        mrz_line_2_results = regex_mrz_line_2.match(mrz.upper())
        if mrz_line_2_results:
            return mrz_line_2_results.groupdict()
        return None

        