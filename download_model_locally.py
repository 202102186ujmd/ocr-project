"""
Script para descargar el modelo Paligemma desde HuggingFace Hub
y guardarlo en disco local para uso 100% offline.

Uso:
    python download_model_locally.py

Requiere un token de HuggingFace con acceso al modelo.
Puedes generarlo en: https://huggingface.co/settings/tokens
"""

import os
import sys

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODEL_ID = "google/paligemma-3b-pt-448"

# Ruta local donde se guardará el modelo (debe coincidir con config.py)
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "paligemma-3b-pt-448")

# Token de HuggingFace (leer desde variable de entorno o solicitarlo)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print(
            "⚠️  No se encontró HF_TOKEN en las variables de entorno.\n"
            "   Puedes exportarlo antes de ejecutar este script:\n"
            "       export HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "   O editarlo directamente en este archivo.\n"
        )
        sys.exit(1)

    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    print(f"📥 Descargando modelo '{MODEL_ID}' desde HuggingFace...")
    print(f"📂 Destino: {LOCAL_MODEL_PATH}\n")

    try:
        print("🔄 Descargando procesador...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

        print("🔄 Descargando modelo (esto puede tardar varios minutos)...")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID, token=HF_TOKEN
        )

        print("\n💾 Guardando en disco local...")
        processor.save_pretrained(LOCAL_MODEL_PATH)
        model.save_pretrained(LOCAL_MODEL_PATH)

        print(f"\n✅ Modelo guardado exitosamente en: {LOCAL_MODEL_PATH}")
        print("   Ya puedes iniciar la API con: uvicorn main:app --reload")

    except Exception as e:
        print(f"\n❌ Error durante la descarga: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
