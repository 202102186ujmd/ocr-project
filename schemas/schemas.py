from pydantic import BaseModel
from typing import Optional


class OCRRequest(BaseModel):
    """Modelo para solicitud de OCR"""
    image_url: Optional[str] = None
    description: str = "Solicitud de OCR de imagen"


class OCRResponse(BaseModel):
    """Modelo para respuesta de OCR"""
    success: bool
    text: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class ImageUploadResponse(BaseModel):
    """Modelo para respuesta de carga de imagen"""
    success: bool
    filename: str
    file_path: str
    message: str


class HealthResponse(BaseModel):
    """Modelo para respuesta de salud de la API"""
    status: str
    version: str
    message: str

