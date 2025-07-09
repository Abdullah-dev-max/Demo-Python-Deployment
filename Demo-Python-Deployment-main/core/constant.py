import os
import sys
from enum import Enum
from pathlib import Path


import torch


class Constant:
    EMBEDDING_MODEL_PATH =  Path(__file__).parent.parent / "model" / "facial_recognition_weights.pth" #os.path.join(os.path.dirname(sys.argv[0]), "model", "Smart_One.pth")
    EMBEDDING_DIM: int = 512

    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceVerificationConstant(Enum):
    IMAGE_RESIZE = 112

