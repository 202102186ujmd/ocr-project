from pydantic import BaseModel, Field
from typing import Optional, List


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


# ---------------------------------------------------------------------------
# Document-specific schemas
# ---------------------------------------------------------------------------

class DocumentFieldsResponse(BaseModel):
    """Respuesta genérica de extracción de campos de documento de identidad"""
    document_type: str = Field(..., description="Tipo de documento: 'dui' o 'license'")
    full_name: Optional[str] = None
    id_number: Optional[str] = None
    license_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    expiration_date: Optional[str] = None
    class_category: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    raw_extraction: Optional[str] = None
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")


class DUIResponse(BaseModel):
    """Respuesta de extracción de campos de un DUI de El Salvador"""
    apellidos: Optional[str] = Field(None, description="Apellidos del titular")
    nombres: Optional[str] = Field(None, description="Nombres del titular")
    genero: Optional[str] = Field(None, description="Género del titular")
    fecha_nacimiento: Optional[str] = Field(None, description="Fecha de nacimiento")
    lugar_nacimiento: Optional[str] = Field(None, description="Lugar de nacimiento")
    numero_dui: Optional[str] = Field(None, description="Número único de identidad (9 dígitos)")
    raw_extraction: Optional[str] = None
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    valid: bool = Field(False, description="True si el número DUI tiene formato válido")


class DriverLicenseResponse(BaseModel):
    """Respuesta de extracción de campos de una licencia de conducir de El Salvador"""
    nombre: Optional[str] = Field(None, description="Nombre completo del titular")
    numero_licencia: Optional[str] = Field(None, description="Número de licencia")
    dui: Optional[str] = Field(None, description="Número de DUI asociado")
    clase_categoria: Optional[str] = Field(None, description="Clase/categoría (A, B, C, D…)")
    fecha_vencimiento: Optional[str] = Field(None, description="Fecha de vencimiento")
    genero: Optional[str] = Field(None, description="Género del titular")
    raw_extraction: Optional[str] = None
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    valid: bool = Field(False, description="True si los datos de la licencia son válidos")


class DocumentValidationResponse(BaseModel):
    """Respuesta de validación de la integridad de un documento"""
    document_type: str
    is_valid: bool = Field(..., description="True si el documento pasó todas las validaciones")
    errors: List[str] = Field(default_factory=list, description="Lista de errores de validación")
    warnings: List[str] = Field(default_factory=list, description="Avisos no bloqueantes")
    fields: Optional[dict] = Field(None, description="Campos extraídos usados en la validación")


class BatchProcessResponse(BaseModel):
    """Respuesta de procesamiento por lotes"""
    total: int = Field(..., description="Total de imágenes enviadas")
    successful: int = Field(..., description="Imágenes procesadas exitosamente")
    failed: int = Field(..., description="Imágenes que fallaron")
    results: List[dict] = Field(default_factory=list, description="Resultados por imagen")
    processing_time: float = Field(..., description="Tiempo total de procesamiento en segundos")


