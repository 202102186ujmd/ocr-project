import logging
import os
import time
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import settings
from models.ocr_model import get_ocr_model
from schemas.schemas import (
    BatchProcessResponse,
    DocumentFieldsResponse,
    DocumentValidationResponse,
    DriverLicenseResponse,
    DUIResponse,
    ImageUploadResponse,
    OCRResponse,
)
from utils.helpers import (
    clean_ocr_text,
    delete_file,
    parse_date,
    parse_name,
    save_upload_file,
    validate_dui_fields,
    validate_image,
    validate_license_fields,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR"])


# ---------------------------------------------------------------------------
# Existing endpoints (preserved)
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Carga una imagen para procesamiento OCR.

    - **file**: Archivo de imagen a cargar.

    Returns:
        ImageUploadResponse con información del archivo guardado.
    """
    try:
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido. Permitidos: {', '.join(allowed_types)}",
            )

        result = save_upload_file(file)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

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
    Extrae texto de una imagen usando Paligemma.

    - **image_path**: Ruta de la imagen (relativa o absoluta).
    - **prompt**: Pregunta o instrucción para el modelo (opcional).

    Returns:
        OCRResponse con el texto extraído.
    """
    try:
        validation = validate_image(image_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["message"])

        ocr_model = get_ocr_model()

        if not prompt:
            prompt = "¿Qué texto ves en esta imagen? Extrae todo el texto visible."

        result = ocr_model.extract_text_with_confidence(image_path)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Error al extraer texto")
            )

        return OCRResponse(
            success=True,
            text=result["text"],
            confidence=result["confidence"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en extracción OCR: {str(e)}")
        return OCRResponse(success=False, error=f"Error al procesar imagen: {str(e)}")


@router.post("/extract-from-url", response_model=OCRResponse)
async def extract_text_from_url(image_url: str, prompt: Optional[str] = None):
    """
    Extrae texto de una imagen desde URL.

    - **image_url**: URL de la imagen.
    - **prompt**: Pregunta o instrucción para el modelo (opcional).

    Returns:
        OCRResponse con el texto extraído.
    """
    try:
        import io
        import tempfile
        import urllib.request
        from urllib.parse import urlparse

        from PIL import Image

        # Validate URL scheme to prevent SSRF attacks
        parsed = urlparse(image_url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(
                status_code=400,
                detail="Solo se permiten URLs con esquema http o https",
            )

        logger.info(f"Descargando imagen desde: {image_url}")
        with urllib.request.urlopen(image_url) as response:  # noqa: S310
            image_data = response.read()

        try:
            Image.open(io.BytesIO(image_data)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="URL no contiene una imagen válida")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_data)
            temp_path = tmp_file.name

        try:
            ocr_model = get_ocr_model()

            if not prompt:
                prompt = "¿Qué texto ves en esta imagen? Extrae todo el texto visible."

            result = ocr_model.extract_text_with_confidence(temp_path)

            if not result["success"]:
                raise HTTPException(
                    status_code=500, detail=result.get("error", "Error al extraer texto")
                )

            return OCRResponse(
                success=True,
                text=result["text"],
                confidence=result["confidence"],
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en extracción desde URL: {str(e)}")
        return OCRResponse(success=False, error=f"Error al procesar URL: {str(e)}")


@router.get("/health")
async def health_check():
    """Verifica el estado del servicio OCR."""
    try:
        ocr_model = get_ocr_model()
        return {
            "status": "healthy",
            "model_loaded": ocr_model is not None,
            "model_name": settings.paligemma_model,
            "device": settings.device,
        }
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(status_code=500, detail="Servicio no disponible")


# ---------------------------------------------------------------------------
# New document-specific endpoints
# ---------------------------------------------------------------------------

@router.post("/extract-dui", response_model=DUIResponse)
async def extract_dui(file: UploadFile = File(...)):
    """
    Procesa un DUI de El Salvador y extrae campos estructurados.

    - **file**: Imagen del DUI.

    Returns:
        DUIResponse con apellidos, nombres, género, fecha de nacimiento,
        lugar de nacimiento y número único de identidad.
    """
    start_time = time.time()

    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Permitidos: {', '.join(allowed_types)}",
        )

    saved = save_upload_file(file)
    if not saved["success"]:
        raise HTTPException(status_code=400, detail=saved["message"])

    file_path = saved["file_path"]
    try:
        validation = validate_image(file_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["message"])

        ocr_model = get_ocr_model()
        fields = ocr_model.extract_dui_fields(file_path)

        is_valid, _errors, _warnings = validate_dui_fields(fields)

        return DUIResponse(
            apellidos=parse_name(fields.get("apellidos")),
            nombres=parse_name(fields.get("nombres")),
            genero=fields.get("genero"),
            fecha_nacimiento=parse_date(fields.get("fecha_nacimiento")),
            lugar_nacimiento=fields.get("lugar_nacimiento"),
            numero_dui=fields.get("numero_dui"),
            raw_extraction=clean_ocr_text(fields.get("raw_extraction", "")),
            processing_time=round(time.time() - start_time, 3),
            valid=is_valid,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en extract-dui: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar DUI: {str(e)}")
    finally:
        delete_file(file_path)


@router.post("/extract-license", response_model=DriverLicenseResponse)
async def extract_license(file: UploadFile = File(...)):
    """
    Procesa una licencia de conducir de El Salvador y extrae campos estructurados.

    - **file**: Imagen de la licencia.

    Returns:
        DriverLicenseResponse con nombre, número de licencia, DUI,
        clase/categoría, fecha de vencimiento y género.
    """
    start_time = time.time()

    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Permitidos: {', '.join(allowed_types)}",
        )

    saved = save_upload_file(file)
    if not saved["success"]:
        raise HTTPException(status_code=400, detail=saved["message"])

    file_path = saved["file_path"]
    try:
        validation = validate_image(file_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["message"])

        ocr_model = get_ocr_model()
        fields = ocr_model.extract_license_fields(file_path)

        is_valid, _errors, _warnings = validate_license_fields(fields)

        return DriverLicenseResponse(
            nombre=parse_name(fields.get("nombre")),
            numero_licencia=fields.get("numero_licencia"),
            dui=fields.get("dui"),
            clase_categoria=fields.get("clase_categoria"),
            fecha_vencimiento=parse_date(fields.get("fecha_vencimiento")),
            genero=fields.get("genero"),
            raw_extraction=clean_ocr_text(fields.get("raw_extraction", "")),
            processing_time=round(time.time() - start_time, 3),
            valid=is_valid,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en extract-license: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error al procesar licencia: {str(e)}"
        )
    finally:
        delete_file(file_path)


@router.post("/validate-document", response_model=DocumentValidationResponse)
async def validate_document(
    file: UploadFile = File(...),
    document_type: str = "dui",
):
    """
    Procesa y valida la integridad estructural de un documento de identidad.

    - **file**: Imagen del documento.
    - **document_type**: Tipo de documento: ``'dui'`` o ``'license'``.

    Returns:
        DocumentValidationResponse indicando si el documento es válido,
        junto con los errores y advertencias encontrados.
    """
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Permitidos: {', '.join(allowed_types)}",
        )

    if document_type not in ("dui", "license"):
        raise HTTPException(
            status_code=400,
            detail="document_type debe ser 'dui' o 'license'",
        )

    saved = save_upload_file(file)
    if not saved["success"]:
        raise HTTPException(status_code=400, detail=saved["message"])

    file_path = saved["file_path"]
    try:
        validation = validate_image(file_path)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["message"])

        ocr_model = get_ocr_model()

        if document_type == "dui":
            fields = ocr_model.extract_dui_fields(file_path)
            is_valid, errors, warnings = validate_dui_fields(fields)
        else:
            fields = ocr_model.extract_license_fields(file_path)
            is_valid, errors, warnings = validate_license_fields(fields)

        return DocumentValidationResponse(
            document_type=document_type,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            fields=fields,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en validate-document: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error al validar documento: {str(e)}"
        )
    finally:
        delete_file(file_path)


@router.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process(files: List[UploadFile] = File(...)):
    """
    Procesa múltiples imágenes de documentos en lote.

    - **files**: Lista de imágenes a procesar.

    Returns:
        BatchProcessResponse con los resultados individuales y estadísticas.
    """
    start_time = time.time()

    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    results: list = []
    successful = 0
    failed = 0

    ocr_model = get_ocr_model()

    for file in files:
        item_start = time.time()
        if file.content_type not in allowed_types:
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": "Tipo de archivo no permitido",
                }
            )
            failed += 1
            continue

        saved = save_upload_file(file)
        if not saved["success"]:
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": saved["message"],
                }
            )
            failed += 1
            continue

        file_path = saved["file_path"]
        try:
            validation = validate_image(file_path)
            if not validation["valid"]:
                results.append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "error": validation["message"],
                    }
                )
                failed += 1
                continue

            result = ocr_model.extract_text_with_confidence(file_path)
            if result["success"]:
                results.append(
                    {
                        "filename": file.filename,
                        "success": True,
                        "text": clean_ocr_text(result.get("text", "")),
                        "confidence": result.get("confidence"),
                        "processing_time": round(time.time() - item_start, 3),
                    }
                )
                successful += 1
            else:
                results.append(
                    {
                        "filename": file.filename,
                        "success": False,
                        "error": result.get("error", "Error desconocido"),
                    }
                )
                failed += 1
        except Exception as e:
            logger.error(f"Error procesando {file.filename}: {str(e)}")
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                }
            )
            failed += 1
        finally:
            delete_file(file_path)

    return BatchProcessResponse(
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
        processing_time=round(time.time() - start_time, 3),
    )


