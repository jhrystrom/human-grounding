import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_main_dir(name: str = "data", create_dir: bool = True) -> Path:
    path = Path(__file__).parent.parent.parent / name
    if not path.exists() and create_dir:
        path.mkdir()
    return path


DATA_DIR = get_main_dir(name="data")
PLOT_DIR = get_main_dir(name="plots")
CACHE_DIR = get_main_dir(name=".cache")
OUTPUT_DIR = get_main_dir(name="output")
IMAGE_DIR = get_main_dir(name="images")
MODEL_DIR = Path(os.getenv("MODEL_DIR", ".models"))
