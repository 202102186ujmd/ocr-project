import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


def save_upload_file(upload_file: UploadFile, destination: str = None) -> dict:
    """
    Guarda un archivo subido en el directorio de uploads

    Args:
        upload_file: Archivo subido
        destination: Directorio de destino (opcional)

    Returns:
        Diccionario con información del archivo guardado
    """
    try:
        if destination is None:
            destination = settings.upload_dir

        # Crear directorio si no existe
        os.makedirs(destination, exist_ok=True)

        # Generar nombre de archivo único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{upload_file.filename}"
        file_path = os.path.join(destination, filename)

        # Guardar archivo
        with open(file_path, "wb") as buffer:
            buffer.write(upload_file.file.read())

        logger.info(f"Archivo guardado: {file_path}")

        return {
            "success": True,
            "filename": filename,
            "file_path": file_path,
            "message": "Archivo guardado exitosamente"
        }

    except Exception as e:
        logger.error(f"Error al guardar archivo: {str(e)}")
        return {
            "success": False,
            "filename": None,
            "file_path": None,
            "message": f"Error al guardar archivo: {str(e)}"
        }


def validate_image(file_path: str, max_size: int = None) -> dict:
    """
    Valida que un archivo sea una imagen válida

    Args:
        file_path: Ruta del archivo
        max_size: Tamaño máximo permitido en bytes

    Returns:
        Diccionario con resultado de validación
    """
    try:
        max_size = max_size or settings.max_upload_size

        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            return {"valid": False, "message": "Archivo no encontrado"}

        # Verificar tamaño
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return {"valid": False, "message": f"Archivo muy grande. Máximo: {max_size} bytes"}

        # Verificar que es una imagen válida
        with Image.open(file_path) as img:
            img.verify()

        logger.info(f"Imagen validada: {file_path}")
        return {"valid": True, "message": "Imagen válida"}

    except Exception as e:
        logger.error(f"Error al validar imagen: {str(e)}")
        return {"valid": False, "message": f"Imagen inválida: {str(e)}"}


def get_file_size(file_path: str) -> dict:
    """Obtiene el tamaño de un archivo en diferentes formatos"""
    try:
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024

        return {
            "bytes": size_bytes,
            "kb": round(size_kb, 2),
            "mb": round(size_mb, 2)
        }
    except Exception as e:
        logger.error(f"Error al obtener tamaño del archivo: {str(e)}")
        return None


def delete_file(file_path: str) -> dict:
    """Elimina un archivo"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Archivo eliminado: {file_path}")
            return {"success": True, "message": "Archivo eliminado exitosamente"}
        else:
            return {"success": False, "message": "Archivo no encontrado"}
    except Exception as e:
        logger.error(f"Error al eliminar archivo: {str(e)}")
        return {"success": False, "message": f"Error al eliminar: {str(e)}"}


def clean_uploads_directory(days_old: int = 7) -> dict:
    """
    Elimina archivos más antiguos que X días del directorio de uploads

    Args:
        days_old: Número de días

    Returns:
        Diccionario con resultado de la limpieza
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

