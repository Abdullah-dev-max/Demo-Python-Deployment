import re
from google.cloud import vision
import os
import numpy as np
import base64
import cv2
from PIL import Image
from PIL.Image import Image as BaseImage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'sonic-potion-417909-4df3012dc349.json'

client = vision.ImageAnnotatorClient()



def detect_text(image):
    """Detects text in an image using Google Vision API OCR."""
    # Convert NumPy array to bytes (Google Vision API expects bytes)
    _, encoded_image = cv2.imencode('.jpg', image)
    image_bytes = encoded_image.tobytes()

    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description  # Return extracted text
    else:
        return None  # No text detected


def extract_face(image):
    """Detects and extracts the face from an image and returns it as a NumPy array."""
    _, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()
    

    image = vision.Image(content=content)
    face_response = client.face_detection(image=image)

    if not face_response.face_annotations:
        return None

    # Read the image with OpenCV
    image_cv = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    # Get face bounding box
    face = face_response.face_annotations[0]
    vertices = face.bounding_poly.vertices
    x1, y1 = vertices[0].x, vertices[0].y
    x2, y2 = vertices[2].x, vertices[2].y

    # Crop the face
    face_image = image_cv[y1:y2, x1:x2]
    face_image = cv2.resize(face_image, (112, 112))

    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    return face_pil


def extract_document_details(ocr_text):
    """Extracts key details from CNIC, Passport, or Driver's License dynamically, only accepting English."""
    if not ocr_text:
        return {"Error": "No text detected in the image"}

    # Filter out non-English characters
    ocr_text = re.sub(r'[^a-zA-Z0-9\s.,:/-]', '', ocr_text)
    extracted_info = {}

    # Normalize text (remove extra spaces and newlines)
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]

    # Identify document type
    document_type = "Unknown"
    if "National Identity Card" in ocr_text or "CNIC" in ocr_text or "NICOP" in ocr_text:
        document_type = "CNIC"
    elif "Passport" or "PASSPORT" or "passport" in ocr_text:
        document_type = "Passport"
    elif "Driver's License" in ocr_text or "Driving License" in ocr_text or "DL" in ocr_text:
        document_type = "Driver's License"

    extracted_info["Document Type"] = document_type

    # Common Fields
    extracted_info["Name"] = "Not Found"
    extracted_info["Date of Birth"] = "Not Found"
    extracted_info["Gender"] = "Not Found"
    extracted_info["Nationality"] = "Not Found"
    extracted_info["Country of Residence"] = "Not Found"
    extracted_info["Date of Issue"] = "Not Found"
    extracted_info["Date of Expiry"] = "Not Found"

    print(lines)

    # Extract Name
    extracted_info["Name"] = get_name(lines, document_type)

    # Extract Date of Birth
    extracted_info["Date of Birth"] = get_dob(lines, document_type)

    # Extract Gender
    extracted_info["Gender"] = get_gender(lines, document_type)

    # Extract Nationality
    extracted_info["Nationality"] = get_nationality(lines, document_type)

    # Extract Date of Issue
    extracted_info["Date of Issue"] = get_issue_date(lines, document_type)

    # Extract Date of Expiry
    extracted_info["Date of Expiry"] = get_expiry_date(lines, document_type)

    # Extract Country of Residence
    for i, line in enumerate(lines):
        if "Country of Stay" in line and i + 2 < len(lines):
            next_line = lines[i + 2].strip()
            if len(next_line) > 2 and next_line.isalpha():
                extracted_info["Country of Residence"] = next_line
                break

    # Extract Document-Specific Fields
    if document_type == "CNIC":
        extracted_info["CNIC Number"] = "Not Found"
        cnic_match = re.search(r"\b\d{5}-\d{7}-\d{1}\b", ocr_text)
        if cnic_match:
            extracted_info["CNIC Number"] = cnic_match.group(0)
    elif document_type == "Passport":
        extracted_info["Passport Number"] = "Not Found"
        passport_match = re.search(r"\b[A-Z]{2}\d{7}\b", ocr_text)
        print(passport_match)
        if passport_match:
            extracted_info["Passport Number"] = passport_match.group(0)
    elif document_type == "Driver's License":
        extracted_info["License Number"] = "Not Found"
        license_match = re.search(r"\b[A-Z0-9]{5,15}\b", ocr_text)
        if license_match:
            extracted_info["License Number"] = license_match.group(0)

    return extracted_info

def get_name(lines, document_type) -> str:
    def is_valid_name(name: str) -> bool:
        return any(char.isalpha() for char in name)
    
    for i, line in enumerate(lines):
        if document_type == "CNIC":
            if "Name" in line and i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if is_valid_name(candidate):
                    return candidate
        elif document_type == "Passport":
            for i, line in enumerate(lines):
                if "Given Name" in line or "Given Names" in line:
            # The actual name is often after 'Given Names' or two lines after
                    if i + 1 < len(lines) and len(lines[i + 1].strip()) > 2:
                        candidate = lines[i + 1].strip()
                    elif i + 2 < len(lines):
                        candidate =  lines[i + 2].strip()
                    if is_valid_name(candidate):
                        return candidate
            if "Nationality" in line and i - 1 < len(lines):
                return lines[i - 1].strip()
        elif document_type == "Driver's License":
            if "Name" in line and i + 1 < len(lines):
                return lines[i + 1].strip()

    return "Not Found"

def get_dob(lines, document_type) -> str:
    for i, line in enumerate(lines):
        if document_type == "CNIC":
            if "Date of Birth" in line and i + 1 < len(lines):
                dob_match = re.search(r"\d{2}[-./]\d{2}[-./]\d{4}", lines[i + 1])
                return dob_match.group(0) if dob_match else "Not Found"
        elif document_type == "Passport":
            if "Date of Birth" in line and i + 1 < len(lines):
                dob_match = re.search(r"\d{2}\s\w{3}\s\d{4}", " ".join(lines))
                return dob_match.group(0) if dob_match else "Not Found"
        elif document_type == "Driver's License":
            if "Date of Birth" in line and i + 1 < len(lines):
                dob_match = re.search(r"\d{2}[-./]\d{2}[-./]\d{4}", lines[i + 1])
                return dob_match.group(0) if dob_match else "Not Found"

    return "Not Found"

def get_gender(lines, document_type) -> str:
    for i, line in enumerate(lines):
        if document_type == "CNIC":
            if "Gender" in line and i + 1 < len(lines):
                gender_text = lines[i + 1].strip()
                return "Male" if "M" in gender_text else "Female" if "F" in gender_text else "Unknown"
        elif document_type == "Passport":
            if "Sex" in line and i + 1 < len(lines):
                gender_text = lines[i + 1].strip()
                return "Male" if "M" in gender_text else "Female" if "F" in gender_text else "Unknown"
        elif document_type == "Driver's License":
            if "Sex" in line and i + 1 < len(lines):
                gender_text = lines[i + 1].strip()
                return "Male" if "M" in gender_text else "Female" if "F" in gender_text else "Unknown"

    return "Not Found"

def get_nationality(lines, document_type) -> str:
    for i, line in enumerate(lines):
        if document_type == "Passport":
            if "Nationality" in line and i + 1 < len(lines):
                return lines[i + 1].strip()

    return "Not Found"

def get_issue_date(lines, document_type) -> str:
    for i, line in enumerate(lines):
        if document_type == "Passport":
            if "Date of Issue" in line or "Issue Date" in line:
                for offset in range(1, 3):  # Look in next 2 lines
                    if i + offset < len(lines):
                        dob_match = re.search(r"\d{2}\s\w{3}\s\d{4}", lines[i + offset])
                        if dob_match:
                            return dob_match.group(0)
        elif document_type == "CNIC":
            if "Date of Issue" in line and i + 1 < len(lines):
                dob_match = re.search(r"\d{2}[-./]\d{2}[-./]\d{4}", lines[i + 1])
                return dob_match.group(0) if dob_match else "Not Found"
        elif document_type == "Driver's License":
            if "Date of Issue" in line and i + 1 < len(lines):
                dob_match = re.search(r"\d{2}[-./]\d{2}[-./]\d{4}", lines[i + 1])
                return dob_match.group(0) if dob_match else "Not Found"
    return "Not Found"

def get_expiry_date(lines, document_type) -> str:
    for i, line in enumerate(lines):
        if document_type == "Passport":
            if "Date of Expiry" in line or "Expiry Date" in line:
                for offset in range(1, 3):  # Look in next 2 lines
                    if i + offset < len(lines):
                        dob_match = re.search(r"\d{2}\s\w{3}\s\d{4}", lines[i + offset])
                        if dob_match:
                            return dob_match.group(0)
        elif document_type == "CNIC":
            if "Date of Expiry" in line and i + 1 < len(lines):
                dob_match = re.search(r"\d{2}[-./]\d{2}[-./]\d{4}", lines[i + 1])
                return dob_match.group(0) if dob_match else "Not Found"
        elif document_type == "Driver's License":
            if "Date of Expiry" in line and i + 1 < len(lines):
                dob_match = re.search(r"\d{2}[-./]\d{2}[-./]\d{4}", lines[i + 1])
                return dob_match.group(0) if dob_match else "Not Found"
    return "Not Found"
