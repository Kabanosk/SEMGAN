import logging
from pathlib import Path


def create_logger(log_file_path: Path = Path("logfile.log")) -> logging.Logger:
    """Create a logger instance.

    Args:
        log_file_path (Path): Path to the log file.

    Returns:
        logging.Logger: Logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
