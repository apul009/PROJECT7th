from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB = str(BASE_DIR / "database.db")
SECRET_KEY = "anpr_secret_2024"

EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_ENABLED = False

LABEL_MAP_PATH = str(BASE_DIR / "label_map.pkl")
MODEL_PATH = str(BASE_DIR / "model.pkl")
UPLOAD_DIR = str(BASE_DIR / "static" / "uploads")
VIDEO_DIR = str(BASE_DIR / "uploads")
