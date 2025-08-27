"""
config.py — small helper to load environment variables (e.g., API keys)
"""
import os
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv()

def get_env(key: str, default: str = "") -> str:
    """
    Reads environment variable safely.
    """
    return os.getenv(key, default)
