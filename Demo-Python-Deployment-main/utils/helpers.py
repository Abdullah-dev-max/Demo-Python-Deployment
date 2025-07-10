import logging
import os
from datetime import datetime
from typing import Union

def parse_date(date_str: str) -> Union[datetime, None]:
    """
    Converts a date from YYMMDD format to DD-MM-YYYY format.

    Parameters
    ----------
    yymmdd : str
        Date string in yymmdd, dd.mm.yyyy or YYYY-mm-dd format (e.g., '250526', '1986-08-23' or "13.09.1997' etc.).

    Returns
    -------
    datetime | None
        Return a `datetime` object if the parsing is successfull otherwise return `None`
    """
    try:
        if len(date_str) == 6 and date_str.isdigit():
            # Format: yymmdd
            return datetime.strptime(date_str, "%y%m%d")
        elif "." in date_str and len(date_str) == 10:
            # Format: dd.mm.yyyy
            return datetime.strptime(date_str, "%d.%m.%Y")
        elif "-" in date_str and len(date_str) == 10:
            # Format YYYY-mm-dd
            return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        return


def setup_logger(log_file='logs/app_log.txt'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()  # Get root logger
    logger.setLevel(logging.INFO)

    # Clear existing handlers (important if run multiple times)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(thread)d - %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler (optional)
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
        