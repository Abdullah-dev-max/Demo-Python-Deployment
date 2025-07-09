import gc
import os
from typing import Union

import numpy as np
from PIL import Image
import torch
from PIL.Image import Image as BaseImage
from PIL import ImageEnhance, ImageOps
from torchvision import transforms
import mediapipe as mp
from insightface.app import FaceAnalysis

from model import face_process
from core import FaceVerificationConstant, Constant
from mtlface.modules import MTLFace
import uuid

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
mp_face_mesh = mp.solutions.face_mesh


class FaceRecognizer:
    """
    Example
    -------
        >>> from facial_recognition import FaceRecognizer
        # Converting the deatected from Numpy Array to Pillow Image. THe image size should be `112 x 112`
        >>> img = Image.fromarray(np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8))
        >>> transformed_img = FaceRecognizer.transform_face(img) # Transform needed to process faces
        >>> new_embedding = FaceRecognizer.generate_embedding_of_single_image(transformed_img) # Generate embedding of single image
        >>> images_to_match = torch.stack([torch.Tensor(new_embedding), torch.Tensor(old_embedding)]) # Embeddings of faces to match
        >>> similarity_between_faces = FaceRecognizer.match_faces(images_to_match) # Must in `2 x N` shape. Return the distance between the  two faces. Range: 0 to 1
    """
  
    @staticmethod
    def _load_embedding_model():
        """
        Load Embeddings Model
        """
        if not os.path.exists(Constant.EMBEDDING_MODEL_PATH):
            raise FileNotFoundError(f"Embedding model not found at {Constant.EMBEDDING_MODEL_PATH }")
        _model = MTLFace()
        _model.load_state_dict(torch.load(Constant.EMBEDDING_MODEL_PATH))
        _model.to(Constant.get_device())
        _model.eval()

        return _model

    FACE_DETECTOR = FaceAnalysis(name='buffalo_l', providers=['cpu'])
    FACE_DETECTOR.prepare(ctx_id=-1)
    EMBEDDING_MODE = _load_embedding_model()

    @classmethod
    def detect_face(cls, image: Image.Image) -> Union[Image.Image, None]:
        """
        Detects and crops face using MediaPipe FaceMesh.
        Falls back to MTCNN, then manual CNIC crop.
        """
        def preprocess_for_id_photo(image: Image.Image) -> Image.Image:
            image = image.convert("RGB")
            image = ImageOps.exif_transpose(image)
            image = ImageOps.autocontrast(image)
            image = ImageEnhance.Sharpness(image).enhance(2.0)
            return image
    
        image_rgb = preprocess_for_id_photo(image.convert("RGB"))
        img_np = np.array(image_rgb)
        h, w, _ = img_np.shape

        # Try FaceMesh
        # print("[INFO] Trying MediaPipe FaceMesh...")
        # with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        #     results = face_mesh.process(img_np)
        #     if results.multi_face_landmarks:
        #         print("[INFO] FaceMesh detected face.")
        #         face_landmarks = results.multi_face_landmarks[0].landmark
        #         xs = [int(lm.x * w) for lm in face_landmarks]
        #         ys = [int(lm.y * h) for lm in face_landmarks]
        #         xmin, xmax = max(0, min(xs)), min(w, max(xs))
        #         ymin, ymax = max(0, min(ys)), min(h, max(ys))
        #         pad = 10
        #         xmin, ymin = max(xmin - pad, 0), max(ymin - pad, 0)
        #         xmax, ymax = min(xmax + pad, w), min(ymax + pad, h)
        #         face_crop =image_rgb.crop((xmin, ymin, xmax, ymax))
        #         import matplotlib.pyplot as plt
        #         plt.imsave("cropeed_img.jpg", np.array(face_crop))
        #         face_crop = face_crop.convert("RGB").resize((112, 112))
        #         return face_crop

        # ------- InsightFace Detector -------------------------------
        # with torch.no_grad():
        #     faces = cls.FACE_DETECTOR.get(img_np)

        # if faces:
        #     print("[INFO] Face detected using InsightFace.")
        #     face = faces[0]
        #     aligned_face = None
        #     try:
        #         # Try alignment
        #         if hasattr(face, "crop_align") and callable(face.crop_align):
        #             aligned_face = face.crop_align(img_np)
        #         else:
        #             print("[INFO] Falling back to bbox crop.")
        #             try:
        #                 x1, y1, x2, y2 = face.bbox.astype(int)
        #                 face_crop = img_np[y1:y2, x1:x2]
        #                 aligned_face = Image.fromarray(face_crop).convert("RGB").resize((112, 112))
        #             except Exception as e:
        #                 print(f"[ERROR] Bounding box crop also failed: {e}")
        #                 aligned_face = None

        #     except Exception as e:
        #         print(f"[WARN] Failed: {e}")

        #     if aligned_face is not None:
        #         face_pil = Image.fromarray(aligned_face).convert("RGB").resize((112, 112)) if isinstance(aligned_face, np.ndarray) else aligned_face.convert("RGB").resize((112, 112))
        #         return face_pil

        # Fallback to MTCNN
        print("[INFO] FaceMesh failed, trying MTCNN...")
        try:
            with torch.no_grad():
                face = face_process(image_rgb)
            return face
        except Exception as e:
            print(f"[ERROR] MTCNN failed: {e}")
        # Fallback to manual CNIC crop or manual CNIC crop
        # print("[INFO] Both methods failed, falling back to CNIC crop...")
        # return FaceRecognizer.crop_face_from_cnic(image)

    @staticmethod
    def crop_face_from_cnic(image: Image.Image) -> Union[Image.Image, None]:
        """
        Manually crops the CNIC photo region based on known layout.
        """
        width, height = image.size
        left = int(width * 0.05)
        top = int(height * 0.55)
        right = int(width * 0.45)
        bottom = int(height * 0.95)
        print("[INFO] Manually cropped CNIC face region.")
        face_crop = image.crop((left, top, right, bottom))
        
        print("[DEBUG] CNIC crop size:", face_crop.size)
        face_crop = face_crop.convert("RGB").resize((112, 112))
        print("[DEBUG] CNIC crop new size:", face_crop.size)
        print("[DEBUG] CNIC crop mode:", face_crop.mode)
        print("[DEBUG] CNIC crop type:", type(face_crop))

        return face_crop
    
    @classmethod
    def transform_face(cls, img: BaseImage) -> "torch.Tensor":
        """
        Transform faces and normalize them

        Parameters
        ----------
            img : PIL.Image
                shape [C, W, H]

        Returns
        -------
            torch.Tensor
        """
        transform = transforms.Compose([
            transforms.Resize(FaceVerificationConstant.IMAGE_RESIZE.value),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        ])
        input_img = transform(img).unsqueeze(0).to(Constant.get_device())

        return input_img

    @classmethod
    def generate_embedding_of_single_image(cls, img) -> "torch.Tensor":
        """
        Generate embedding of single image

        Parameters
        ----------
            img : torch.Tensor
                shape [C, W, H]

        Returns
        -------
            torch.Tensor
        """
        with torch.no_grad():
            x_vec = cls.EMBEDDING_MODE.encode(img)

        x_vec = x_vec.cpu()

        return x_vec[0]

    @classmethod
    def match_faces(cls, images: torch.Tensor) -> float:
        """
            Calculate the similarity between faces

            Parameters
            ----------
                images : list
                    list of images. each image is a torch tensor. Shape [B, C, W, H]. `B` is batch

            Returns
            -------
            float
        """
        with torch.no_grad():
            x_vec = cls.EMBEDDING_MODE.encode(images)

        x_vec = x_vec.cpu()

        similarity_between_faces = cls.calculate_cosine_similarity(x_vec)

        return similarity_between_faces

    @staticmethod
    def calculate_cosine_similarity(x_vec: Union["np.array", "torch.Tensor"]) -> float:
        """
        Calculate the cosine similarity between two vectors

        Cosine similarity is defined as:

        cosine_similarity = (A · B) / (||A|| * ||B||)

        where:
            A · B   = dot product of vectors A and B
            ||A||   = magnitude (Euclidean norm) of vector A
            ||B||   = magnitude (Euclidean norm) of vector B

        Parameters
        ----------
            x_vec : Union["np.array", "torch.Tensor"]
                shape [2, 512]

        Returns
        -------
            float: Cosine similarity between vec1 and vec2 (range: -1 to 1)
        """
        similarity_point = np.dot(x_vec[0], x_vec[1]) / (np.linalg.norm(x_vec[0]) * np.linalg.norm(x_vec[1]))

        return similarity_point
