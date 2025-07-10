import gc
import logging
import os
import traceback
from pathlib import Path
from typing import Union

import numpy as np
import requests
import torch
import tqdm
from PIL import Image
from PIL import ImageEnhance, ImageOps
from PIL.Image import Image as BaseImage
from insightface.app import FaceAnalysis
from torchvision import transforms

from core import FaceVerificationConstant, Constant
from model import face_process
from mtlface.modules import MTLFace


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
            try:
                os.makedirs(os.path.dirname(Constant.EMBEDDING_MODEL_PATH), exist_ok=True)
                # Streaming, so we can iterate over the response.
                response = requests.get(Constant.EMBEDDING_MODEL_URL, stream=True, timeout=15)

                if response.status_code != 200:
                    raise requests.HTTPError(f"Failed to download embedding model: {response.status_code}")
                # Sizes in bytes.
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                logging.info(f"Downloading embedding model...")
                with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                    with open(Constant.EMBEDDING_MODEL_PATH, "wb") as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)

                if total_size != 0 and progress_bar.n != total_size:
                    raise RuntimeError("Could not download file")
                logging.info(f"Embedding model saved to {Constant.EMBEDDING_MODEL_PATH}")
            except requests.exceptions.Timeout:
                logging.error(f"Request timed out while downloading the model weights..."
                              f"\nError Details {traceback.format_exc()}")
                Path(Constant.EMBEDDING_MODEL_PATH).unlink(missing_ok=True)
                raise requests.exceptions.Timeout("Request timed out while downloading the model weights...")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to download embedding model: {str(e)}"
                              f"\nError Details {traceback.format_exc()}")
                Path(Constant.EMBEDDING_MODEL_PATH).unlink(missing_ok=True)
                raise requests.exceptions.RequestException(f"An error occurred: {str(e)}")
            except Exception as e:
                logging.error(f"Failed to download embedding model: {str(e)}"
                              f"\nError Details {traceback.format_exc()}")
                Path(Constant.EMBEDDING_MODEL_PATH).unlink(missing_ok=True)
                raise Exception(f"Failed to download embedding model: {str(e)}")
        logging.info(f"Loading embedding model...")
        _model = MTLFace()
        _model.load_state_dict(torch.load(Constant.EMBEDDING_MODEL_PATH))
        _model.to(Constant.get_device())
        logging.info(f"Embedding model loaded on {Constant.get_device()}.")
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
        logging.info("Trying MTCNN...")
        try:
            with torch.no_grad():
                face = face_process(image_rgb)
            return face
        except Exception as e:
            logging.info(f"[ERROR] MTCNN failed: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

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
        logging.info("Manually cropped CNIC face region.")
        face_crop = image.crop((left, top, right, bottom))
        
        logging.debug("CNIC crop size:", face_crop.size)
        face_crop = face_crop.convert("RGB").resize((112, 112))
        logging.debug("CNIC crop new size:", face_crop.size)
        logging.debug("CNIC crop mode:", face_crop.mode)
        logging.debug("CNIC crop type:", type(face_crop))

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
        try:
            with torch.no_grad():
                x_vec = cls.EMBEDDING_MODE.encode(img)

            x_vec = x_vec.cpu()

            return x_vec[0]
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

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
        try:
            with torch.no_grad():
                x_vec = cls.EMBEDDING_MODE.encode(images)

            x_vec = x_vec.cpu()

            similarity_between_faces = cls.calculate_cosine_similarity(x_vec)

            return similarity_between_faces
        except Exception as e:
            raise Exception(f"Failed to match faces: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()

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
