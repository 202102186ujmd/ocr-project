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
    def __init__(self):
        self.device = settings.device
        self._load_model()

    def _load_model(self):
        """Carga el modelo desde disco local"""
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
                torch_dtype=torch.float16
            ).to(self.device)

            self.model.eval()
            logger.info("✅ Modelo cargado exitosamente desde disco local")

        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {str(e)}")
            raise

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

