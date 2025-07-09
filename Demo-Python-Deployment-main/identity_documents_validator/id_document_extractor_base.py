from abc import ABC, abstractmethod
from typing import Union

from core.data_contract import IdentityDocument

from PIL import Image

class BaseIDExtractor(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    # @classmethod
    def get_text_from_image(self, b64_image_string: str) -> Union[None, "IdentityDocument"]:
        """
        Parameters
        ----------
        image : str
            base-64 encoded image
        
        Returns
        -------
        An object of `IdentityDocument` containing information extarcted from the identity document
        """
        pass