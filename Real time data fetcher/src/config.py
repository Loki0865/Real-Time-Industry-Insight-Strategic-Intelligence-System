import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

def get_env(key: str, default=None):
    return os.getenv(key, default)
