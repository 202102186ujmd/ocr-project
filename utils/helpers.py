import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import UploadFile
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def save_upload_file(upload_file: UploadFile, destination: str = None) -> dict:
    """
    Guarda un archivo subido en el directorio de uploads.

    Args:
        upload_file: Archivo subido.
        destination: Directorio de destino (opcional).

    Returns:
        Diccionario con información del archivo guardado.
    """
    try:
        if destination is None:
            destination = settings.upload_dir

        os.makedirs(destination, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{upload_file.filename}"
        file_path = os.path.join(destination, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(upload_file.file.read())

        logger.info(f"Archivo guardado: {file_path}")

        return {
            "success": True,
            "filename": filename,
            "file_path": file_path,
            "message": "Archivo guardado exitosamente",
        }

    except Exception as e:
        logger.error(f"Error al guardar archivo: {str(e)}")
        return {
            "success": False,
            "filename": None,
            "file_path": None,
            "message": f"Error al guardar archivo: {str(e)}",
        }


def validate_image(file_path: str, max_size: int = None) -> dict:
    """
    Valida que un archivo sea una imagen válida.

    Args:
        file_path: Ruta del archivo.
        max_size: Tamaño máximo permitido en bytes.

    Returns:
        Diccionario con resultado de validación.
    """
    try:
        max_size = max_size or settings.max_upload_size

        if not os.path.exists(file_path):
            return {"valid": False, "message": "Archivo no encontrado"}

        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return {
                "valid": False,
                "message": f"Archivo muy grande. Máximo: {max_size} bytes",
            }

        with Image.open(file_path) as img:
            img.verify()

        logger.info(f"Imagen validada: {file_path}")
        return {"valid": True, "message": "Imagen válida"}

    except Exception as e:
        logger.error(f"Error al validar imagen: {str(e)}")
        return {"valid": False, "message": f"Imagen inválida: {str(e)}"}


def get_file_size(file_path: str) -> Optional[dict]:
    """Obtiene el tamaño de un archivo en diferentes formatos."""
    try:
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        return {
            "bytes": size_bytes,
            "kb": round(size_kb, 2),
            "mb": round(size_mb, 2),
        }
    except Exception as e:
        logger.error(f"Error al obtener tamaño del archivo: {str(e)}")
        return None


def delete_file(file_path: str) -> dict:
    """Elimina un archivo."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Archivo eliminado: {file_path}")
            return {"success": True, "message": "Archivo eliminado exitosamente"}
        return {"success": False, "message": "Archivo no encontrado"}
    except Exception as e:
        logger.error(f"Error al eliminar archivo: {str(e)}")
        return {"success": False, "message": f"Error al eliminar: {str(e)}"}


def clean_uploads_directory(days_old: int = 7) -> dict:
    """
    Elimina archivos más antiguos que *days_old* días del directorio de uploads.

    Args:
        days_old: Número de días.

    Returns:
        Diccionario con resultado de la limpieza.
    """
    try:
        import time

        current_time = time.time()
        deleted_count = 0

        for filename in os.listdir(settings.upload_dir):
            file_path = os.path.join(settings.upload_dir, filename)
            if os.path.isfile(file_path):
                file_age = (current_time - os.path.getmtime(file_path)) / (24 * 3600)
                if file_age > days_old:
                    os.remove(file_path)
                    deleted_count += 1

        logger.info(f"Limpieza completada. Archivos eliminados: {deleted_count}")
        return {"success": True, "deleted_files": deleted_count}

    except Exception as e:
        logger.error(f"Error al limpiar directorio: {str(e)}")
        return {"success": False, "message": str(e)}


# ---------------------------------------------------------------------------
# OCR text cleaning
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """
    Limpia el texto bruto producido por el modelo OCR.

    - Elimina líneas vacías duplicadas.
    - Normaliza espacios en blanco internos.
    - Elimina caracteres de control no imprimibles.
    """
    if not text:
        return ""
    # Eliminar caracteres de control no imprimibles (salvo \\n, \\t)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\xA0-\uFFFF]", "", text)
    # Normalizar espacios en blanco internos por línea
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.splitlines()]
    # Eliminar líneas vacías consecutivas duplicadas
    cleaned: List[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

def parse_date(raw: Optional[str]) -> Optional[str]:
    """
    Intenta normalizar una cadena de fecha a formato ISO (YYYY-MM-DD).

    Soporta los formatos más comunes encontrados en documentos salvadoreños:
    DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD MM YYYY.
    Retorna la cadena original si no puede parsear.
    """
    if not raw:
        return None
    raw = raw.strip()
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %m %Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # No se pudo parsear – devolver tal cual para no perder información
    return raw


def parse_name(raw: Optional[str]) -> Optional[str]:
    """Normaliza un nombre: capitaliza cada palabra y elimina espacios extras."""
    if not raw:
        return None
    return " ".join(word.capitalize() for word in raw.split())


def parse_id_number(raw: Optional[str]) -> Optional[str]:
    """
    Extrae solo dígitos y guiones de un número de identificación.
    Útil para limpiar OCR con caracteres extra.
    """
    if not raw:
        return None
    return re.sub(r"[^\d\-]", "", raw.strip())


# ---------------------------------------------------------------------------
# Document validators
# ---------------------------------------------------------------------------

def validate_dui_number(numero: Optional[str]) -> bool:
    """
    Valida el formato del número DUI de El Salvador.
    Formato esperado: 8 dígitos + guión + 1 dígito  →  XXXXXXXX-X
    o bien 9 dígitos sin guión.
    """
    if not numero:
        return False
    clean = re.sub(r"\s", "", numero)
    # Aceptar con o sin guión
    return bool(re.fullmatch(r"\d{8}-\d", clean) or re.fullmatch(r"\d{9}", clean))


def validate_license_number(numero: Optional[str]) -> bool:
    """
    Valida que el número de licencia de conducir no esté vacío y contenga
    al menos 4 caracteres alfanuméricos (la estructura exacta puede variar).
    """
    if not numero:
        return False
    clean = re.sub(r"\s", "", numero)
    return bool(re.search(r"[A-Za-z0-9]{4,}", clean))


def validate_expiration_date(fecha: Optional[str]) -> bool:
    """
    Valida que la fecha de vencimiento de la licencia esté en el rango
    esperado (2020-2035) y que no haya expirado antes del año 2000.
    """
    if not fecha:
        return False
    normalized = parse_date(fecha)
    if not normalized:
        return False
    try:
        dt = datetime.strptime(normalized, "%Y-%m-%d")
        return 2020 <= dt.year <= 2035
    except ValueError:
        return False


def validate_dui_fields(fields: dict) -> tuple:
    """
    Valida los campos extraídos de un DUI.

    Returns:
        (is_valid, errors, warnings) – tupla con resultado de validación.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not fields.get("apellidos"):
        errors.append("Apellidos no encontrados")
    if not fields.get("nombres"):
        errors.append("Nombres no encontrados")
    if not fields.get("numero_dui"):
        errors.append("Número de DUI no encontrado")
    elif not validate_dui_number(fields["numero_dui"]):
        errors.append(
            f"Número de DUI inválido: '{fields['numero_dui']}'. "
            "Debe tener 9 dígitos (XXXXXXXX-X)."
        )
    if not fields.get("fecha_nacimiento"):
        warnings.append("Fecha de nacimiento no encontrada")
    if not fields.get("genero"):
        warnings.append("Género no encontrado")

    return len(errors) == 0, errors, warnings


def validate_license_fields(fields: dict) -> tuple:
    """
    Valida los campos extraídos de una licencia de conducir.

    Returns:
        (is_valid, errors, warnings) – tupla con resultado de validación.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not fields.get("nombre"):
        errors.append("Nombre no encontrado")
    if not fields.get("numero_licencia"):
        errors.append("Número de licencia no encontrado")
    elif not validate_license_number(fields["numero_licencia"]):
        errors.append(
            f"Número de licencia inválido: '{fields['numero_licencia']}'"
        )
    if not fields.get("fecha_vencimiento"):
        warnings.append("Fecha de vencimiento no encontrada")
    elif not validate_expiration_date(fields["fecha_vencimiento"]):
        warnings.append(
            f"Fecha de vencimiento fuera de rango esperado: '{fields['fecha_vencimiento']}'"
        )
    if not fields.get("clase_categoria"):
        warnings.append("Clase/categoría no encontrada")

    return len(errors) == 0, errors, warnings


