import os


class Settings:
    # API
    app_name: str = "OCR Document Processing API"
    app_version: str = "2.0.0"

    # Modelo Paligemma - Ruta local
    PALIGEMMA_LOCAL_PATH = os.path.join(
        os.path.dirname(__file__),
        "models/paligemma-3b-pt-448"
    )

    # Modo offline - solo usar archivos locales
    USE_LOCAL_MODEL_ONLY = True

    # GPU / dispositivo
    device: str = os.getenv("DEVICE", "cuda")
    dtype: str = os.getenv("DTYPE", "float16")

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
        "Extract and list all visible fields including: "
        "Apellidos (last name), Nombres (first name), "
        "Género (gender), Fecha de Nacimiento (date of birth), "
        "Lugar de Nacimiento (place of birth), "
        "Número Único de Identidad (9-digit DUI number). "
        "Return each field on its own line in format: Field: Value. "
        "Extract EXACTLY as written on the document."
    )

    # Prompt para licencia de conducir de El Salvador
    license_prompt: str = (
        "This is a driver's license (Licencia de Conducir) from El Salvador. "
        "Extract and list all visible fields including: "
        "Nombre (full name), Número de Licencia (license number), "
        "DUI (identity number), Clase/Categoría (class/category), "
        "Fecha de Vencimiento (expiration date), Género (gender). "
        "Return each field on its own line in format: Field: Value. "
        "Extract EXACTLY as written on the document."
    )


settings = Settings()

# Crear directorio de uploads si no existe
os.makedirs(settings.upload_dir, exist_ok=True)