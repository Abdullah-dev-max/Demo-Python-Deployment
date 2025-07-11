import base64
import json
import logging
import os
import threading
import traceback
import uuid
import warnings

from datetime import datetime
from functools import wraps

warnings.filterwarnings("ignore")

import cv2
import jwt
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, status as HTTPStatus
from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from qdrant_client.models import PointStruct, Distance, VectorParams

load_dotenv()

from utils.helpers import setup_logger

setup_logger()

from core.constant import Constant
from core.data_contract import (
    APIStatus,
    IdentityDocumentType,
    IdentityDocument,
    OCRInboundRequest,
    FaceRegisterInboundRequest,
    FaceDeleteInboundRequest,
    FaceUpdateInboundRequest,
    OCROutboundResponse,
    FaceOutboundResponse,
    CustomException
)
from facial_recognition import FaceRecognizer
from identity_documents_validator.cnic_extractor import CNICExtractor
from identity_documents_validator.passport_extractor import PassportExtractor

MODL_LOCK = threading.Lock()
setup_logger()
logging.info(f"Application started at {datetime.now()}")

QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_PORT = int(os.getenv('QDRANT_PORT'))
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
API_KEY = os.getenv('API_KEY')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', Constant.EMBEDDING_DIM))
SECRET_KEY = os.environ.get("SECRET_KEY", "secret")
uuid_pattern = r"'(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})'"
client = QdrantClient(url=f"https://{QDRANT_HOST}:{QDRANT_PORT}", api_key=API_KEY)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/testing")
async def testing():
    """
    A simple test endpoint that returns "Hello World"
    """
    return {"message": "Hello World"}

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    This endpoint returns the Swagger UI page for the API.
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="FastAPI Swagger UI"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint() -> dict:
    """
    This endpoint returns the OpenAPI schema for the API. The OpenAPI schema is a JSON object that describes the API's endpoints, methods, parameters, and responses. It is used by tools like Swagger UI to generate an interactive API documentation.

    Returns:
        A JSON object containing the OpenAPI schema.
    """
    return app.openapi()

def formatUUID(string: str) -> str:
    """
    Extracts the UUID from a string that contains a UUID in the format "(UUID)".
    """
    start = string.index("(") + 2
    end = string.index(")") - 1
    return string[start:end]


def generate_jwt_token(user_id: str) -> str:
    """
    Generates a JWT token for the given user_id.

    The token is generated using the HS256 algorithm and the SECRET_KEY
    environment variable. The token contains a "user_id" field with the user's ID.
    """
    token = jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm="HS256")
    return token


def decode_jwt_token(token: str) -> dict:
    """
    Decodes a JWT token and returns the payload.

    If the token is invalid or expired, it raises an HTTPException with a 401
    status code and a detail message.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def jwt_token_required(func):
    """
    Decorator to validate JWT token.

    This decorator checks if the request contains a valid JWT token. If the
    token is invalid or expired, it raises an HTTPException with a 401 status
    code and a detail message.
    """
    @wraps(func)
    async def wrapper(request: Request):
        """
        Checks if the request contains a valid JWT token.
        """
        authorization_header = request.headers.get("Authorization")
        if not authorization_header or not authorization_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Bearer token is missing")
        
        token = authorization_header.split(" ")[1]  # Extract the token from the Authorization header
        
        try:
            payload = decode_jwt_token(token)
            user_id = payload.get("user_id")
            request.state.user_id = user_id  # Attach user_id to request state
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Execute the wrapped function and return its result
        return await func(request)  # Return only the result of the wrapped function
    
    return wrapper


@app.middleware("http")
async def validate_large_base64_images(request: Request, call_next):
    """
    Middleware to validate the size of base64-encoded images in incoming requests.

    Args:
        request (Request): The incoming request object.
        call_next: The next middleware or route handler to be called.

    Returns:
        Response: The response from the next middleware or route handler, or an HTTPException if validation fails.
    """
    if request.method in ["POST", "PUT"]:
        # Check content length from headers
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 150 * 1024 * 1024:
            return JSONResponse(content={"detail": "File size exceeds 150MB limit."}, status_code=413)

        try:
            # Read and decode request body
            body = await request.body()
            data = json.loads(body.decode("utf-8"))
            base64_data = data.get("image")
            if base64_data:
                # Decode base64 image and check size
                decoded_data = base64.b64decode(base64_data)
                if len(decoded_data) > 150 * 1024 * 1024:
                    return JSONResponse(content={"detail": "Decoded image exceeds 150MB."}, status_code=413)
        except json.JSONDecodeError:
            return JSONResponse(content={"detail": "Invalid JSON format."}, status_code=400)
        except Exception as e:
            # Handle other exceptions related to request format
            return JSONResponse(content={"detail": f"Invalid request format: {str(e)}"}, status_code=400)

    # Proceed to the next middleware or route handler
    response = await call_next(request)
    return response

# client = QdrantClient("localhost", port=6333)


def decode_base64_image(base64_string):
    """
    Convert a base64-encoded string to an OpenCV image.

    Args:
        base64_string (str): The base64 string representation of the image.

    Returns:
        numpy.ndarray: The decoded image in OpenCV format.

    Raises:
        HTTPException: If the input string is not a valid base64 image format.
    """
    try:
        # Decode the base64 string to binary image data
        image_data = base64.b64decode(base64_string)
        
        # Convert binary data to a NumPy array
        np_arr = np.frombuffer(image_data, np.uint8)
        
        # Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        # Raise HTTPException if decoding fails
        raise HTTPException(status_code=HTTPStatus.HTTP_400_BAD_REQUEST, detail="Invalid base64 image format")

def initialize_qdrant():
    """
    Initializes the Qdrant collection for storing face embeddings.

    Creates a new collection in Qdrant with the specified name and vector configuration.
    """
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,  # The size of the vectors in the collection
                distance=Distance.COSINE,  # The distance metric used for similarity search
            ),
        )
        print(f"Qdrant collection {COLLECTION_NAME} created.")


initialize_qdrant()


def is_face_already_exist(new_embedding, threshold=0.35):

    """
    Checks if a face already exists in the Qdrant collection based on the cosine similarity of embeddings.

    Args:
        new_embedding (numpy.ndarray): The embedding vector to be checked.
        threshold (float): The similarity threshold above which the face is considered to exist.

    Returns:
        bool: True if the face already exists, False otherwise.
        float: The cosine similarity of the closest match if the face exists, 0 otherwise.
        int: The ID of the closest match if the face exists, -1 otherwise.
    """
    print("Checking if face exists in Qdrant collection...")
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=new_embedding.tolist(),
        limit=1,
        with_payload=True,  # Ensure the payload (including vector) is returned
        with_vectors=True   # Make sure the vector is included in the search result
    )
    if not search_result or len(search_result) == 0:
        logging.info("No existing faces found in Qdrant. Treating as new face.")
        return False, 0, -1
    
    if search_result:
        logging.info("\nClosest Matches in Qdrant:")
        for idx, result in enumerate(search_result):
            logging.info(f"{idx + 1}. Face ID: {result.id}, Score: {result.score}")

        best_match = search_result[0]  # Top match
        logging.debug(f"\nBest Match - Face ID: {best_match.id}, Score: {best_match.score}")
    
    if search_result and search_result[0].score > threshold:
        logging.debug(f"Closest match in Qdrant - Face ID: {search_result[0].id}, Score: {search_result[0].score}")
        logging.debug("Face already exists in Qdrant collection.")
        return True, search_result[0].score, search_result[0].id

    logging.info(f"Closest match in Qdrant - Face ID: {search_result[0].id}, Score: {search_result[0].score}")
    logging.info("Face does not exist in Qdrant collection.")

    # If score is in the range [0.60, 0.70], perform secondary cosine similarity check
    if search_result and 0.60 <= search_result[0].score <= 0.70:
        logging.debug("Score is in the range [0.60, 0.70], performing secondary cosine similarity check...")
        stored_embedding = search_result[0].vector  # Get the stored embedding from the result

        if stored_embedding is None:
            logging.error("Error: No vector found in the search result.")
            return False, 0, -1

        # Flatten both embeddings to ensure they have the same shape
        stored_embedding = np.array(stored_embedding).flatten()
        new_embedding = np.array(new_embedding).flatten()


        # Compute cosine similarity
        similarity = FaceRecognizer.calculate_cosine_similarity(torch.Tensor([stored_embedding, new_embedding]))
        logging.info(f"Cosine similarity after secondary check: {similarity}")

        # If similarity is above threshold, it's a face match
        if similarity >= threshold:
            logging.debug("Face match confirmed by secondary check.")
            return True, similarity, search_result[0].id
    logging.info("Face does not exist in Qdrant collection.")
    return False, 0, -1



def append_embedding(new_embedding, user_id):
    """
    Appends a single embedding to the Qdrant collection.

    Args:
        new_embedding (numpy.ndarray): The embedding vector to be added.
    """
    point_id = str(uuid.uuid4())  # Generate a unique ID for the embedding
    payload = {"user_id": user_id}

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=point_id, vector=new_embedding.tolist(), payload=payload),  # Convert embedding to list and add to Qdrant
        ],
    )
    logging.debug(f"Embedding added with ID {point_id}")  # Log the addition of the embedding
    return str(point_id)


@app.post("/register-face/")
async def register_face(request: Request):
    try:
        data = await request.json()
        face_inbound_request = FaceRegisterInboundRequest.model_validate(data)
        base64_image = face_inbound_request.b64_scanned_img
        user_id = face_inbound_request.user_id # Retrieve user_id from request

        face_response = FaceOutboundResponse(status=APIStatus.UNKNOWN, point_id=None, user_id=user_id,
                                             message="No message")
        if not base64_image or user_id is None:
            face_response.status =APIStatus.FAILED
            face_response.message = "Missing 'Base64ScannedImage' or 'UserId' field."
            logging.error(f"{face_response.message} Point id: {user_id}")
            return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)

        image = decode_base64_image(base64_image)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with MODL_LOCK:
            face_crop = FaceRecognizer.detect_face(pil_image)
        if face_crop is None:
            face_response.status = APIStatus.FAILED
            face_response.message = "No face detected. Please try again."
            logging.error(f"{face_response.message} Point id: {user_id}")
            return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)
        logging.info(f'Face detected for user: {face_response.user_id}')

        transformed_face = FaceRecognizer.transform_face(face_crop)
        logging.info(f'Transformed face for user: {face_response.user_id}')
        with MODL_LOCK:
            embedding = FaceRecognizer.generate_embedding_of_single_image(transformed_face)
        logging.info(f'Generated embedding for user: {face_response.user_id}')

        exists, score, face_id = is_face_already_exist(embedding.cpu().numpy())
        face_response.point_id = str(face_id)
        logging.debug(f"Face Match - Exists: {exists}, Score: {score}, Face ID: {face_response.point_id}")  # Debugging log

        if exists:
            face_response.status = APIStatus.FAILED
            face_response.message = "Face already registered"
            logging.error(f"{face_response.message} User id: {user_id}. Pont Id: {face_response.point_id}")
            return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)

        point_id = append_embedding(embedding.cpu().numpy(), user_id=user_id)
        face_response.point_id = point_id
        logging.info(f"New Face Stored - Point ID: {face_response.point_id}")
        face_response.status = APIStatus.SUCCESS
        face_response.message = "Face registered successfully"
        return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                            status_code=HTTPStatus.HTTP_200_OK)
    except Exception as e:
        error_in_face_registration = CustomException(error=e.__class__.__name__,
                                                     status_code=HTTPStatus.HTTP_500_INTERNAL_SERVER_ERROR,
                                                     message=f"An error occurred while registering face: {str(e)}",
                                                     details=traceback.format_exc())
        logging.error(f"Error in face registration. Error: {error_in_face_registration.message}\n"
                      f"Details: {error_in_face_registration.details}")
        return JSONResponse(content=error_in_face_registration.model_dump(by_alias=True, exclude_none=True, mode="json"),
                            status_code=error_in_face_registration.status_code)


@app.post("/extract-info/")
async def extract_info(request: Request):
    try:
        data = await request.json()
        ocr_inbound_request = OCRInboundRequest.model_validate(data)
        extracted_image = ocr_inbound_request.b64_scanned_img
        point_id = ocr_inbound_request.point_id
        user_id = ocr_inbound_request.user_id
        document_type = ocr_inbound_request.document_type

        ocr_response = OCROutboundResponse(point_id=point_id, user_id=user_id, status=APIStatus.UNKNOWN,
                                           message=f"No message.")

        # Check if the image_base64 field is present
        if not extracted_image:
            ocr_response.status = APIStatus.FAILED
            ocr_response.message = "Missing 'Base64ScannedImage' field."
            logging.error(f"{ocr_response.message} against point id: {ocr_response.point_id}")
            return JSONResponse(content=ocr_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)

        image = decode_base64_image(extracted_image)
        user_info = IdentityDocument()

        if document_type is IdentityDocumentType.CNIC:
            user_info = CNICExtractor.get_text_from_image(b64_image_string=extracted_image)
        elif document_type is IdentityDocumentType.PASSPORT:
            user_info = PassportExtractor.get_text_from_image(b64_image_string=extracted_image)

        ocr_response.data = user_info
        ocr_response.status = APIStatus.UNKNOWN
        logging.info(f"Fetched user info. Point id: {ocr_response.point_id}")

        result = client.retrieve(
            collection_name = COLLECTION_NAME,
            ids=[str(point_id)],
            with_vectors=True,
            with_payload=True
        )

        if not result:
            ocr_response.status = APIStatus.FAILED
            ocr_response.message = f"No entry found in the vector DB for point_id: {point_id}"
            logging.error(ocr_response.message)
            return JSONResponse(content=ocr_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)
        
        vector_db_search_result = result[0]
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with MODL_LOCK:
            face_fetch = FaceRecognizer.detect_face(pil_image)

        if face_fetch is None:
            ocr_response.status = APIStatus.FAILED
            ocr_response.message = "No face detected. Please try again."
            logging.error(f"{ocr_response.message} against point id: {ocr_response.point_id}")
            return JSONResponse(content=ocr_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)

        logging.info(f'Detected face against point id: {ocr_response.point_id}')
        
        face_transformed = FaceRecognizer.transform_face(face_fetch)
        logging.debug(f'Transformed face against point id: {ocr_response.point_id}')
        with MODL_LOCK:
            new_embedding = FaceRecognizer.generate_embedding_of_single_image(face_transformed)
        logging.debug(f'Generated embedding against point id: {ocr_response.point_id}')

        stored_embedding = vector_db_search_result.vector

        similarity = FaceRecognizer.calculate_cosine_similarity([stored_embedding, new_embedding.cpu().numpy()])

        if similarity >= 0.35:
            ocr_response.status = APIStatus.SUCCESS
            ocr_response.message = f"Face match with similarity {similarity:.2f}"
            logging.info(f"{ocr_response.message}. Point ID: {ocr_response.point_id}")
            return JSONResponse(content=ocr_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)
        else:
            ocr_response.status = APIStatus.FAILED
            ocr_response.message = f"Face does not match. Similarity: {similarity:.2f}"
            logging.error(f"{ocr_response.message}. Point ID: {ocr_response.point_id}")
            return JSONResponse(content=ocr_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)
        
    except Exception as e:
        error_in_ocr = CustomException(error=e.__class__.__name__, status_code=HTTPStatus.HTTP_500_INTERNAL_SERVER_ERROR,
                                       message=f"An error occurred in OCR: {str(e)}", details=traceback.format_exc())
        logging.error(f"Error in face registration. Error: {error_in_ocr.message}\nDetails: {error_in_ocr.details}")
        return JSONResponse(content=error_in_ocr.model_dump(by_alias=True, exclude_none=True, mode="json"),
                            status_code=error_in_ocr.status_code)


@app.delete("/delete-face/")
async def delete_face(request: Request):
    try:
        data = await request.json()
        face_delete_inbound = FaceDeleteInboundRequest.model_validate(data)
        point_id = face_delete_inbound.point_id
        user_id = face_delete_inbound.user_id

        face_response = FaceOutboundResponse(status=APIStatus.UNKNOWN, point_id=point_id, user_id=user_id,
                                             message="No message.")

        result = client.retrieve(
            collection_name = COLLECTION_NAME,
            ids=[str(point_id)],
            with_vectors=True,
            with_payload=True
        )
        if not result:
            face_response.status = APIStatus.FAILED
            face_response.message = f"No entry found in the vector DB for point_id: {point_id}"
            logging.error(face_response.message)
            return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)
        
        vector_db_search_result = result[0]
        logging.info(f"Point ID found in Vector DB: {vector_db_search_result.id}")
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=[str(vector_db_search_result.id)])
        )
        face_response.status = APIStatus.SUCCESS
        face_response.message = f"Face deleted successfully"
        logging.info(f"{face_response.message}. Point ID: {face_response.point_id}")
        return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                            status_code=HTTPStatus.HTTP_200_OK)
    except Exception as e:
        error_in_face_deletion = CustomException(error=e.__class__.__name__,
                                                 status_code=HTTPStatus.HTTP_500_INTERNAL_SERVER_ERROR,
                                                 message=f"An error occurred while registering face: {str(e)}",
                                                 details=traceback.format_exc())
        logging.error(f"Error in face registration. Error: {error_in_face_deletion.message}\n"
                      f"Details: {error_in_face_deletion.details}")
        return JSONResponse(content=error_in_face_deletion.model_dump(by_alias=True, exclude_none=True, mode="json"),
                            status_code=error_in_face_deletion.status_code)


@app.put("/update-face/")
async def update_face(request: Request):
    try:
        data = await request.json()
        face_update = FaceUpdateInboundRequest.model_validate(data)
        user_id = face_update.user_id
        point_id = face_update.point_id
        extracted_image = face_update.b64_scanned_img

        face_response = FaceOutboundResponse(status=APIStatus.UNKNOWN, point_id=point_id, user_id=user_id,
                                             message="No message.")
        if not extracted_image:
            face_response.status = APIStatus.FAILED
            face_response.message = "Missing 'image_base64' field."
            logging.error(f"{face_response.message} Point id: {face_update.point_id}")
            return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)

        result = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[str(point_id)],
            with_vectors=True,
            with_payload=True
        )

        if not result:
            face_response.status = APIStatus.FAILED
            face_response.message = f"No entry found in the vector DB for point_id: {point_id}"
            logging.error(f"{face_response.message} Point id: {face_update.point_id}")
            return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)

        vector_db_search_result = result[0]
        logging.info(f"Point ID found in Vector DB: {vector_db_search_result.id}")
        logging.info(f"Updating the Vector for Point ID: {vector_db_search_result.id}")

        image = decode_base64_image(extracted_image)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        face_crop = FaceRecognizer.detect_face(pil_image)

        if face_crop is None:
            face_response.status = APIStatus.FAILED
            face_response.message = "No face detected. Please try again."
            logging.error(f"{face_response.message} Point id: {face_update.point_id}")
            return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                status_code=HTTPStatus.HTTP_200_OK)
        logging.info('Detected face')

        face_transformed = FaceRecognizer.transform_face(face_crop)
        logging.info('Transformed face')
        with MODL_LOCK:
            new_embedding = FaceRecognizer.generate_embedding_of_single_image(face_transformed)
        logging.info('Generated embedding')

        new_embedding_list = new_embedding.tolist()

        # 🧠 Check for duplicate in vector DB (excluding same point_id)
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=new_embedding_list,
            limit=1,
            with_payload=True
        )

        if search_result:
            top_match = search_result[0]
            matched_id = str(top_match.id)
            similarity_score = top_match.score

            if matched_id != str(point_id) and similarity_score >= 0.35:
                face_response.status = APIStatus.FAILED
                face_response.message = f"A similar face already exists in the database with point_id: {matched_id}."
                logging.error(f"{face_response.message} Point id: {face_update.point_id}")
                return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                                    status_code=HTTPStatus.HTTP_200_OK)

        # If not duplicate, update the vector
        point = PointStruct(
            id=str(point_id),
            vector=new_embedding_list,
            payload={"user_id": user_id} if user_id else None
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        face_response.status = APIStatus.SUCCESS
        face_response.message = "Face updated successfully."
        logging.info(f"Face updated successfully for Point ID: {face_response.point_id}")
        return JSONResponse(content=face_response.model_dump(by_alias=True, exclude_none=True, mode="json"),
                            status_code=HTTPStatus.HTTP_200_OK)

    except Exception as e:
        error_in_face_update = CustomException(error=e.__class__.__name__,
                                               status_code=HTTPStatus.HTTP_500_INTERNAL_SERVER_ERROR,
                                               message=f"An error occurred while registering face: {str(e)}",
                                               details=traceback.format_exc())
        logging.error(f"Error in face registration. Error: {error_in_face_update.message}\n"
                      f"Details: {error_in_face_update.details}")
        return JSONResponse(content=error_in_face_update.model_dump(by_alias=True, exclude_none=True, mode="json"),
                            status_code=error_in_face_update.status_code)