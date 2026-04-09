import logging
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import os

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_dtype() -> torch.dtype:
    """Resuelve el dtype a usar según la configuración."""
    return torch.float16 if settings.dtype == "float16" else torch.float32


def _parse_fields(raw_text: str) -> dict:
    """
    Parse 'Field: Value' lines returned by the model into a dictionary.
    Lines that don't contain ':' are collected under the special key
    'raw_extraction'.
    """
    result: dict = {}
    raw_lines = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw_lines.append(line)

        if ":" in line:
            key, _, value = line.partition(":")
            key_clean = key.strip().lower().replace(" ", "_")
            result[key_clean] = value.strip()

    result["raw_extraction"] = "\n".join(raw_lines)
    return result


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class OCRModel:
    """Clase para integración del modelo Paligemma."""

    def __init__(self):
        self.device = settings.device
        self.dtype = _resolve_dtype()
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Carga el modelo desde disco local."""
        try:
            # Verificar que el modelo local existe
            if not os.path.exists(settings.PALIGEMMA_LOCAL_PATH):
                raise FileNotFoundError(
                    f"❌ Modelo no encontrado en: {settings.PALIGEMMA_LOCAL_PATH}\n"
                    f"Ejecuta: python download_model_locally.py"
                )

            logger.info(f"📂 Cargando modelo desde: {settings.PALIGEMMA_LOCAL_PATH}")

            # Cargar desde disco local (sin internet)
            self.processor = AutoProcessor.from_pretrained(
                settings.PALIGEMMA_LOCAL_PATH,
                local_files_only=True  # ⭐ Clave: solo archivos locales
            )

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                settings.PALIGEMMA_LOCAL_PATH,
                local_files_only=True,  # ⭐ Clave: solo archivos locales
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info(f"✅ Modelo cargado exitosamente desde disco local en {self.device}")

        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {str(e)}")
            raise

    def _generate_text(self, image: Image.Image, prompt: str) -> str:
        """
        Genera texto a partir de una imagen y un prompt.

        Args:
            image: Imagen PIL
            prompt: Prompt para el modelo

        Returns:
            Texto generado
        """
        try:
            # Preparar inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Generar predicción
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=False
                )

            # Decodificar resultado
            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            return text

        except Exception as e:
            logger.error(f"Error en generación de texto: {str(e)}")
            raise

    def extract_text(self, image_path: str, prompt: str = None) -> str:
        """
        Extrae texto de una imagen usando Paligemma.

        Args:
            image_path: Ruta a la imagen
            prompt: Pregunta o prompt para el modelo (opcional)

        Returns:
            Texto extraído de la imagen
        """
        try:
            if prompt is None:
                prompt = settings.ocr_prompt

            # Cargar imagen
            image = Image.open(image_path).convert("RGB")

            # Generar texto
            text = self._generate_text(image, prompt)
            logger.info(f"✅ Texto extraído de {image_path}")
            return text

        except Exception as e:
            logger.error(f"❌ Error al extraer texto: {str(e)}")
            raise

    def extract_text_with_confidence(self, image_path: str, prompt: str = None) -> dict:
        """
        Extrae texto con confianza estimada.

        Args:
            image_path: Ruta a la imagen
            prompt: Prompt personalizado (opcional)

        Returns:
            Diccionario con texto, confianza y estado
        """
        try:
            text = self.extract_text(image_path, prompt)
            return {
                "text": text,
                "confidence": 0.85,  # Placeholder - Paligemma no proporciona confianza nativa
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ Error en extract_text_with_confidence: {str(e)}")
            return {
                "text": None,
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }

    def extract_dui_fields(self, image_path: str) -> dict:
        """
        Extrae campos específicos de un DUI de El Salvador.

        Args:
            image_path: Ruta a la imagen del DUI

        Returns:
            Diccionario con campos extraídos
        """
        try:
            logger.info(f"📄 Procesando DUI desde: {image_path}")

            # Cargar imagen
            image = Image.open(image_path).convert("RGB")

            # Generar texto con prompt específico para DUI
            raw_text = self._generate_text(image, settings.dui_prompt)

            # Parsear campos
            fields = _parse_fields(raw_text)

            logger.info(f"✅ Campos DUI extraídos: {len(fields)} campos")
            return fields

        except Exception as e:
            logger.error(f"❌ Error al extraer campos DUI: {str(e)}")
            raise

    def extract_license_fields(self, image_path: str) -> dict:
        """
        Extrae campos específicos de una licencia de conducir de El Salvador.

        Args:
            image_path: Ruta a la imagen de la licencia

        Returns:
            Diccionario con campos extraídos
        """
        try:
            logger.info(f"📄 Procesando Licencia desde: {image_path}")

            # Cargar imagen
            image = Image.open(image_path).convert("RGB")

            # Generar texto con prompt específico para licencia
            raw_text = self._generate_text(image, settings.license_prompt)

            # Parsear campos
            fields = _parse_fields(raw_text)

            logger.info(f"✅ Campos de Licencia extraídos: {len(fields)} campos")
            return fields

        except Exception as e:
            logger.error(f"❌ Error al extraer campos de licencia: {str(e)}")
            raise


# ---------------------------------------------------------------------------
# Singleton accessor (lazy loading + in-memory cache)
# ---------------------------------------------------------------------------

_ocr_model: Optional[OCRModel] = None


def get_ocr_model() -> OCRModel:
    """
    Retorna la instancia global del modelo OCR.
    La instancia se carga una sola vez (lazy loading) y se mantiene en memoria.

    Returns:
        Instancia de OCRModel
    """
    global _ocr_model
    if _ocr_model is None:
        logger.info("🔄 Primera carga del modelo OCR...")
        _ocr_model = OCRModel()
    return _ocr_model