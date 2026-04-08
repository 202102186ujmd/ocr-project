import logging
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from config import settings

logger = logging.getLogger(__name__)


class OCRModel:
    """Clase para integración del modelo Paligemma"""

    def __init__(self):
        """Inicializa el modelo Paligemma"""
        self.device = settings.device
        self.model_name = settings.paligemma_model
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carga el modelo y procesador de Paligemma"""
        try:
            logger.info(f"Cargando modelo: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            raise

    def extract_text(self, image_path: str, prompt: str = "¿Qué texto ves en esta imagen?") -> Optional[str]:
        """
        Extrae texto de una imagen usando Paligemma

        Args:
            image_path: Ruta a la imagen
            prompt: Pregunta o prompt para el modelo

        Returns:
            Texto extraído de la imagen
        """
        try:
            # Cargar imagen
            image = Image.open(image_path).convert("RGB")

            # Procesar imagen y prompt
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Generar predicción
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            # Decodificar resultado
            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Texto extraído de {image_path}")
            return text

        except Exception as e:
            logger.error(f"Error al extraer texto: {str(e)}")
            raise

    def extract_text_with_confidence(self, image_path: str) -> dict:
        """
        Extrae texto con confianza estimada

        Args:
            image_path: Ruta a la imagen

        Returns:
            Diccionario con texto y confianza
        """
        try:
            text = self.extract_text(image_path)
            return {
                "text": text,
                "confidence": 0.85,  # Placeholder - Paligemma no proporciona confianza
                "success": True
            }
        except Exception as e:
            return {
                "text": None,
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }


# Instancia global del modelo
ocr_model: Optional[OCRModel] = None


def get_ocr_model() -> OCRModel:
    """Obtiene instancia del modelo OCR"""
    global ocr_model
    if ocr_model is None:
        ocr_model = OCRModel()
    return ocr_model

