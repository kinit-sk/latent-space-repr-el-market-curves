from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.success(f"Loaded config from {config_path}.")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise
