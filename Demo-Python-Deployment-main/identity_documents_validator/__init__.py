from core.data_contract import IdentityDocumentType
from .id_document_extractor_base import BaseIDExtractor
from .cnic_extractor import CNICExtractor
from .passport_extractor import PassportExtractor

def get_document_ectractor(document_type: IdentityDocumentType):
    document_extractor = BaseIDExtractor()
    if document_type is IdentityDocumentType.PASSPORT:
        document_extractor = PassportExtractor()
    elif document_extractor is IdentityDocumentType.CNIC:
        document_extractor = CNICExtractor()
    return document_extractor