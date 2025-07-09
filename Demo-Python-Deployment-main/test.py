import base64
import io
import os
import unittest
from datetime import datetime

from PIL import Image
from dotenv import load_dotenv

from identity_documents_validator.cnic_extractor import CNICExtractor
from identity_documents_validator.passport_extractor import PassportExtractor

load_dotenv()


class TestOCR(unittest.TestCase):
    def test_extract_ocr_from_passport(self):
        img_file_path = os.path.join(os.getcwd(), "e2.jpg")
        img_opened = Image.open(img_file_path).convert("RGB")
        img_buffered = io.BytesIO()
        img_opened.save(img_buffered, format="PNG")
        img_bytes = img_buffered.getvalue()
        encoded_img = base64.b64encode(img_bytes).decode("utf-8")
        passport_meta_data = PassportExtractor.get_text_from_image(b64_image_string=encoded_img)
        self.assertIsNotNone(passport_meta_data)
        self.assertIsInstance(passport_meta_data.name, str)
        self.assertTrue(passport_meta_data.is_expired)
        self.assertIsInstance(passport_meta_data.dob, datetime)
        self.assertIsNotNone(passport_meta_data.passport_number)

    def test_extract_ocr_from_cnic(self):
        img_file_path = os.path.join(os.getcwd(), "cnic_2.jpg")
        img_opened = Image.open(img_file_path).convert("RGB")
        img_buffered = io.BytesIO()
        img_opened.save(img_buffered, format="PNG")
        img_bytes = img_buffered.getvalue()
        encoded_img = base64.b64encode(img_bytes).decode("utf-8")
        cnic_meta_data = CNICExtractor.get_text_from_image(b64_image_string=encoded_img)
        self.assertIsNotNone(cnic_meta_data)
        self.assertIsInstance(cnic_meta_data.name, str)
        self.assertFalse(cnic_meta_data.is_expired)
        self.assertIsInstance(cnic_meta_data.dob, datetime)
        self.assertIsNotNone(cnic_meta_data.cnic_number)


if __name__ == '__main__':
    unittest.main()
