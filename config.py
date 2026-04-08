import os
from typing import Optional


class Settings:
    # Modelo
    MODEL_NAME = "paligemma"
    MODEL_ID = "google/paligemma-3b-pt-448"  # Versión de 3B parámetros

    # GPU
    DEVICE = "cuda"  # o "cpu" si no tienes GPU
    DTYPE = "float16"  # Usar float16 para ahorrar VRAM

    # Directorios
    UPLOAD_DIR = "./uploads"
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

    # OCR
    OCR_PROMPT = "Extract all text from this document. Return structured information."

    # API
    API_TITLE = "OCR Document Processing API"
    API_VERSION = "1.0.0"


settings = Settings()

# Crear directorio de uploads si no existe
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)