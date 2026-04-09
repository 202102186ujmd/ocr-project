import os


class Settings:
    # API
    app_name: str = "OCR Document Processing API"
    app_version: str = "1.0.0"

    # Modelo Paligemma
    paligemma_model: str = os.getenv("PALIGEMMA_MODEL", "google/paligemma-3b-pt-448")
    # Usar ruta local en lugar de descargar de internet
    PALIGEMMA_LOCAL_PATH = os.path.join(
        os.path.dirname(__file__),
        "models/paligemma-3b-pt-448"
    )

    # Modo offline - no descargar de internet
    USE_LOCAL_MODEL_ONLY = True

    # GPU / dispositivo
    device: str = os.getenv("DEVICE", "cuda")
    # "float16" para GPU, "float32" para CPU
    dtype: str = os.getenv("DTYPE", "float16")
    # Habilitar cuantización int8 para reducir VRAM (requiere bitsandbytes)
    use_int8: bool = os.getenv("USE_INT8", "false").lower() == "true"

    # Directorios
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))  # 10 MB

    # Generación de texto
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))

    # Prompts genéricos
    ocr_prompt: str = (
        "Extract all text from this document. Return structured information."
    )

    # Prompt para DUI de El Salvador
    dui_prompt: str = (
        "This is a DUI (Documento Único de Identidad) from El Salvador. "
        "Extract all fields: Apellidos (last name), Nombres (first name), "
        "Género (gender), Fecha de Nacimiento (date of birth), "
        "Lugar de Nacimiento (place of birth), "
        "Número Único de Identidad (9-digit DUI number). "
        "Return each field on its own line as 'Field: Value'."
    )

    # Prompt para licencia de conducir de El Salvador
    license_prompt: str = (
        "This is a driver's license (Licencia de Conducir) from El Salvador. "
        "Extract all fields: Nombre (full name), "
        "Número de Licencia (license number), "
        "DUI (identity document number), "
        "Clase/Categoría (class/category such as A, B, C, D), "
        "Fecha de Vencimiento (expiration date), "
        "Género (gender). "
        "Return each field on its own line as 'Field: Value'."
    )


settings = Settings()

# Crear directorio de uploads si no existe
os.makedirs(settings.upload_dir, exist_ok=True)