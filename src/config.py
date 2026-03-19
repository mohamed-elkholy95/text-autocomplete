"""Configuration."""
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_NGRAM = 4
MIN_NGRAM = 1
TOP_K = 5
MIN_FREQUENCY = 2
CONTEXT_WINDOW = 50
STREAMLIT_THEME = {"primaryColor": "#1f77b4", "backgroundColor": "#0e1117", "secondaryBackgroundColor": "#262730", "textColor": "#ffffff"}
API_HOST = "0.0.0.0"
API_PORT = 8010
