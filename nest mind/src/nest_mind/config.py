from pathlib import Path

class Config:
    PROJECT_NAME = "Nest Mind"
    VERSION = "0.1.0"

    # Add a data directory for context persistence
    DATA_DIR = Path(__file__).parent / "data"
    DATA_DIR.mkdir(exist_ok=True)  # create the folder if it doesn't exist

