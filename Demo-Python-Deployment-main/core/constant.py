import os
import sys
from enum import Enum
from pathlib import Path


import torch


class Constant:
    EMBEDDING_MODEL_PATH = os.path.expanduser("~/Astra/KYC/models/facial_recognition_weights.pth")
    EMBEDDING_MODEL_URL = r"https://astra-python-main-git-lfs-issue.s3.me-central-1.amazonaws.com/KYC/ML-Weights/facial_recognition_weights.pth"
    EMBEDDING_DIM: int = 512

    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceVerificationConstant(Enum):
    IMAGE_RESIZE = 112

