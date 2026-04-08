import logging
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_dtype() -> torch.dtype:
    return torch.float16 if settings.dtype == "float16" else torch.float32


def _parse_fields(raw_text: str) -> dict:
    """
    Parse 'Field: Value' lines returned by the model into a dictionary.
    Lines that don't contain ':' are collected under the special key
    'raw_extraction'.
    """
    result: dict = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip().lower()] = value.strip()
    result["raw_extraction"] = raw_text
    return result


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class OCRModel:
    """Integración del modelo Paligemma con caché y soporte GPU optimizado."""

    def __init__(self):
        self.device = settings.device
        self.model_name = settings.paligemma_model
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[PaliGemmaForConditionalGeneration] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Carga el modelo y procesador de Paligemma con soporte float16 / int8."""
        try:
            logger.info(f"Cargando modelo: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            load_kwargs: dict = {
                "low_cpu_mem_usage": True,
                "device_map": "auto",
            }

            if settings.use_int8:
                # Cuantización int8 – requiere bitsandbytes
                load_kwargs["load_in_8bit"] = True
                logger.info("Cuantización int8 activada")
            else:
                load_kwargs["torch_dtype"] = _resolve_dtype()

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs
            )
            self.model.eval()
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def extract_text(self, image_path: str, prompt: str = "¿Qué texto ves en esta imagen?") -> Optional[str]:
        """
        Extrae texto de una imagen usando Paligemma.

        Args:
            image_path: Ruta a la imagen.
            prompt: Pregunta o prompt para el modelo.

        Returns:
            Texto extraído de la imagen.
        """
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=False,
                )

            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Texto extraído de {image_path}")
            return text

        except Exception as e:
            logger.error(f"Error al extraer texto: {str(e)}")
            raise

    def extract_text_with_confidence(self, image_path: str) -> dict:
        """
        Extrae texto con confianza estimada.

        Returns:
            Diccionario con 'text', 'confidence' y 'success'.
        """
        try:
            text = self.extract_text(image_path)
            return {
                "text": text,
                "confidence": 0.85,  # Paligemma no proporciona probabilidades directas
                "success": True,
            }
        except Exception as e:
            return {
                "text": None,
                "confidence": 0.0,
                "success": False,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Document-specific extraction
    # ------------------------------------------------------------------

    def extract_dui_fields(self, image_path: str) -> dict:
        """
        Extrae campos estructurados de un DUI de El Salvador.

        Returns:
            Diccionario con campos del DUI.
        """
        raw = self.extract_text(image_path, prompt=settings.dui_prompt)
        parsed = _parse_fields(raw)

        return {
            "apellidos": parsed.get("apellidos"),
            "nombres": parsed.get("nombres"),
            "genero": parsed.get("género") or parsed.get("genero"),
            "fecha_nacimiento": (
                parsed.get("fecha de nacimiento")
                or parsed.get("fecha_nacimiento")
            ),
            "lugar_nacimiento": (
                parsed.get("lugar de nacimiento")
                or parsed.get("lugar_nacimiento")
            ),
            "numero_dui": (
                parsed.get("número único de identidad")
                or parsed.get("numero único de identidad")
                or parsed.get("numero_dui")
            ),
            "raw_extraction": raw,
        }

    def extract_license_fields(self, image_path: str) -> dict:
        """
        Extrae campos estructurados de una licencia de conducir de El Salvador.

        Returns:
            Diccionario con campos de la licencia.
        """
        raw = self.extract_text(image_path, prompt=settings.license_prompt)
        parsed = _parse_fields(raw)

        return {
            "nombre": parsed.get("nombre"),
            "numero_licencia": (
                parsed.get("número de licencia")
                or parsed.get("numero de licencia")
                or parsed.get("numero_licencia")
            ),
            "dui": parsed.get("dui"),
            "clase_categoria": (
                parsed.get("clase/categoría")
                or parsed.get("clase/categoria")
                or parsed.get("clase_categoria")
            ),
            "fecha_vencimiento": (
                parsed.get("fecha de vencimiento")
                or parsed.get("fecha_vencimiento")
            ),
            "genero": parsed.get("género") or parsed.get("genero"),
            "raw_extraction": raw,
        }

    def extract_document_fields(self, image_path: str) -> dict:
        """
        Extrae campos genéricos de un documento de identidad.

        Returns:
            Diccionario con los campos encontrados.
        """
        extraction_prompt = (
            "Extract the following information from this ID/License document:\n"
            "- Full Name\n"
            "- ID Number / License Number\n"
            "- Date of Birth\n"
            "- Expiration Date\n"
            "- Class/Category\n"
            "- Gender\n"
            "- Address (if visible)\n"
            "Return each field on its own line as 'Field: Value'."
        )
        raw = self.extract_text(image_path, prompt=extraction_prompt)
        parsed = _parse_fields(raw)

        return {
            "full_name": parsed.get("full name"),
            "id_number": parsed.get("id number"),
            "license_number": parsed.get("license number"),
            "date_of_birth": parsed.get("date of birth"),
            "expiration_date": parsed.get("expiration date"),
            "class_category": parsed.get("class/category"),
            "gender": parsed.get("gender"),
            "address": parsed.get("address"),
            "raw_extraction": raw,
        }


# ---------------------------------------------------------------------------
# Singleton accessor (lazy loading + in-memory cache)
# ---------------------------------------------------------------------------

_ocr_model: Optional[OCRModel] = None


def get_ocr_model() -> OCRModel:
    """Retorna la instancia global del modelo OCR (cargada una sola vez)."""
    global _ocr_model
    if _ocr_model is None:
        _ocr_model = OCRModel()
    return _ocr_model

