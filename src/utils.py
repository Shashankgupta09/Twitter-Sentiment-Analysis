import os
import logging

def get_logger(name="sentiment_analysis", log_file="training.log"):
    """
    Creates and returns a logger that outputs to both standard output and a file.
    """
    # Ensure logs directory or parent directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if logger is already configured
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        
        # File Handler
        f_handler = logging.FileHandler(log_file, encoding='utf-8')
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        
    return logger

def ensure_dir(path):
    """
    Ensures that a directory exists, creating it if necessary.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False
