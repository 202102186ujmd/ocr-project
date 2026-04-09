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


def _map_fields(parsed: dict, field_mapping: dict) -> dict:
    """
    Maps parsed field keys to target keys using a mapping dictionary.

    Args:
        parsed: Dictionary of parsed 'field: value' pairs (keys lowercased).
        field_mapping: Dict mapping target_key -> list of possible source keys.

    Returns:
        Dictionary with target keys and matched values (None if not found).
    """
    fields: dict = {}
    for target_key, source_keys in field_mapping.items():
        for src in source_keys:
            if src in parsed:
                fields[target_key] = parsed[src]
                break
        else:
            fields[target_key] = None
    return fields


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
                torch_dtype=_resolve_dtype()
            ).to(self.device)

            self.model.eval()
            logger.info("✅ Modelo cargado exitosamente desde disco local")

        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {str(e)}")
            raise

    def _generate_text(self, image_path: str, prompt: str) -> str:
        """
        Genera texto a partir de una imagen y un prompt usando Paligemma.

        Args:
            image_path: Ruta a la imagen.
            prompt: Instrucción para el modelo.

        Returns:
            Texto generado por el modelo (sin incluir el prompt de entrada).
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens (skip the input prompt tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        return self.processor.decode(generated_tokens, skip_special_tokens=True)

    def extract_text(self, image_path: str, prompt: Optional[str] = None) -> dict:
        """
        Extrae texto genérico de una imagen.

        Args:
            image_path: Ruta a la imagen.
            prompt: Prompt personalizado (opcional).

        Returns:
            Diccionario con 'success', 'text' y opcionalmente 'error'.
        """
        try:
            if not prompt:
                prompt = settings.ocr_prompt
            text = self._generate_text(image_path, prompt)
            return {"success": True, "text": text}
        except Exception as e:
            logger.error(f"Error en extract_text: {str(e)}")
            return {"success": False, "text": None, "error": str(e)}

    def extract_text_with_confidence(
        self, image_path: str, prompt: Optional[str] = None
    ) -> dict:
        """
        Extrae texto de una imagen e incluye una métrica de confianza.

        La confianza se estima como la media de las probabilidades máximas
        de los tokens generados (entre 0.0 y 1.0).

        Args:
            image_path: Ruta a la imagen.
            prompt: Prompt personalizado (opcional).

        Returns:
            Diccionario con 'success', 'text', 'confidence' y opcionalmente 'error'.
        """
        try:
            if not prompt:
                prompt = settings.ocr_prompt

            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[0][input_len:]
            text = self.processor.decode(generated_ids, skip_special_tokens=True)

            # Estimate confidence from the max probability of each generated token
            confidence = None
            if outputs.scores:
                probs = [
                    torch.softmax(score[0], dim=-1).max().item()
                    for score in outputs.scores
                ]
                if probs:
                    confidence = round(sum(probs) / len(probs), 4)

            return {"success": True, "text": text, "confidence": confidence}

        except Exception as e:
            logger.error(f"Error en extract_text_with_confidence: {str(e)}")
            return {"success": False, "text": None, "confidence": None, "error": str(e)}

    def extract_dui_fields(self, image_path: str) -> dict:
        """
        Extrae los campos de un DUI (Documento Único de Identidad)
        de El Salvador.

        Args:
            image_path: Ruta a la imagen del DUI.

        Returns:
            Diccionario con los campos del DUI:
            apellidos, nombres, genero, fecha_nacimiento,
            lugar_nacimiento, numero_dui, raw_extraction.
        """
        try:
            raw_text = self._generate_text(image_path, settings.dui_prompt)
            parsed = _parse_fields(raw_text)

            # Map Spanish field names from the model response to expected keys
            field_mapping = {
                "apellidos": ["apellidos", "apellido", "last name", "lastname"],
                "nombres": ["nombres", "nombre", "first name", "firstname", "names"],
                "genero": ["género", "genero", "gender", "sexo"],
                "fecha_nacimiento": [
                    "fecha de nacimiento", "fecha_nacimiento",
                    "date of birth", "nacimiento",
                ],
                "lugar_nacimiento": [
                    "lugar de nacimiento", "lugar_nacimiento",
                    "place of birth", "lugar",
                ],
                "numero_dui": [
                    "número único de identidad", "numero_dui", "numero dui",
                    "número dui", "dui", "id number",
                ],
            }

            fields = _map_fields(parsed, field_mapping)
            fields["raw_extraction"] = raw_text
            return fields

        except Exception as e:
            logger.error(f"Error en extract_dui_fields: {str(e)}")
            return {"raw_extraction": "", "error": str(e)}

    def extract_license_fields(self, image_path: str) -> dict:
        """
        Extrae los campos de una licencia de conducir de El Salvador.

        Args:
            image_path: Ruta a la imagen de la licencia.

        Returns:
            Diccionario con los campos de la licencia:
            nombre, numero_licencia, dui, clase_categoria,
            fecha_vencimiento, genero, raw_extraction.
        """
        try:
            raw_text = self._generate_text(image_path, settings.license_prompt)
            parsed = _parse_fields(raw_text)

            # Map Spanish field names from the model response to expected keys
            field_mapping = {
                "nombre": ["nombre", "full name", "name", "nombres"],
                "numero_licencia": [
                    "número de licencia", "numero_licencia", "numero licencia",
                    "license number", "licencia",
                ],
                "dui": [
                    "dui", "id number", "identity document number", "número dui",
                ],
                "clase_categoria": [
                    "clase/categoría", "clase_categoria", "clase categoria",
                    "clase", "categoría", "category", "class",
                ],
                "fecha_vencimiento": [
                    "fecha de vencimiento", "fecha_vencimiento",
                    "expiration date", "vencimiento",
                ],
                "genero": ["género", "genero", "gender", "sexo"],
            }

            fields = _map_fields(parsed, field_mapping)
            fields["raw_extraction"] = raw_text
            return fields

        except Exception as e:
            logger.error(f"Error en extract_license_fields: {str(e)}")
            return {"raw_extraction": "", "error": str(e)}


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

