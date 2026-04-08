import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional

from config import settings
from schemas.schemas import OCRResponse, ImageUploadResponse
from models.ocr_model import get_ocr_model
from utils.helpers import save_upload_file, validate_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR"])


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Carga una imagen para procesamiento OCR

    - **file**: Archivo de imagen a cargar

    Returns:
        ImageUploadResponse con información del archivo guardado
    """
    try:
        # Validar tipo de archivo
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido. Permitidos: {', '.join(allowed_types)}"
            )

        # Guardar archivo
        result = save_upload_file(file)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

        # Validar imagen
        validation = validate_image(result["file_path"])

        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["message"])

        return ImageUploadResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {str(e)}")


@router.post("/extract", response_model=OCRResponse)
async def extract_text(image_path: str, prompt: Optional[str] = None):
    """
    Extrae texto de una imagen usando Paligemma

    - **image_path**: Ruta de la imagen (relativa o absoluta)
    - **prompt**: Pregunta o instrucción para el modelo (opcional)

    Returns:
        OCRResponse con el texto extraído
    """
    try:
        # Validar que la imagen existe
        validation = validate_image(image_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["message"])

        # Obtener modelo OCR
        ocr_model = get_ocr_model()

        # Establecer prompt por defecto si no se proporciona
        if not prompt:
            prompt = "¿Qué texto ves en esta imagen? Extrae todo el texto visible."

        # Extraer texto
        result = ocr_model.extract_text_with_confidence(image_path)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Error al extraer texto"))

        return OCRResponse(
            success=True,
            text=result["text"],
            confidence=result["confidence"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en extracción OCR: {str(e)}")
        return OCRResponse(
            success=False,
            error=f"Error al procesar imagen: {str(e)}"
        )


@router.post("/extract-from-url", response_model=OCRResponse)
async def extract_text_from_url(image_url: str, prompt: Optional[str] = None):
    """
    Extrae texto de una imagen desde URL

    - **image_url**: URL de la imagen
    - **prompt**: Pregunta o instrucción para el modelo (opcional)

    Returns:
        OCRResponse con el texto extraído
    """
    try:
        import urllib.request
        from PIL import Image
        import io

        # Descargar imagen
        logger.info(f"Descargando imagen desde: {image_url}")
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()

        # Validar que es una imagen
        try:
            Image.open(io.BytesIO(image_data)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="URL no contiene una imagen válida")

        # Guardar temporalmente
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_data)
            temp_path = tmp_file.name

        try:
            # Obtener modelo OCR
            ocr_model = get_ocr_model()

            # Establecer prompt por defecto
            if not prompt:
                prompt = "¿Qué texto ves en esta imagen? Extrae todo el texto visible."

            # Extraer texto
            result = ocr_model.extract_text_with_confidence(temp_path)

            if not result["success"]:
                raise HTTPException(status_code=500, detail=result.get("error", "Error al extraer texto"))

            return OCRResponse(
                success=True,
                text=result["text"],
                confidence=result["confidence"]
            )

        finally:
            # Eliminar archivo temporal
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en extracción desde URL: {str(e)}")
        return OCRResponse(
            success=False,
            error=f"Error al procesar URL: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Verifica el estado del servicio OCR"""
    try:
        ocr_model = get_ocr_model()
        return {
            "status": "healthy",
            "model_loaded": ocr_model is not None,
            "model_name": settings.paligemma_model,
            "device": settings.device
        }
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(status_code=500, detail="Servicio no disponible")

