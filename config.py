import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# API Endpoints
FOOTBALL_API_BASE_URL = "https://api.football-data.org/v4"
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# Model Settings
MODEL_PATH = "models/prediction_model.joblib"

# Render optimization settings
MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB max cache
CACHE_DURATION = 3600  # 1 hour
MAX_DB_SIZE = 200 * 1024 * 1024  # 200MB max database
CLEANUP_INTERVAL = 86400  # Clean temporary files daily
