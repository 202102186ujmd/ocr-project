import os
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# 🔑 TU TOKEN AQUÍ
HF_TOKEN = "Cambiar por el token "

# Configurar ubicación local
LOCAL_MODEL_PATH = "./models/paligemma-3b-pt-448"
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

print("📥 Descargando modelo...")

processor = AutoProcessor.from_pretrained(
    "google/paligemma-3b-pt-448",
    token=HF_TOKEN
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-448",
    token=HF_TOKEN
)

print("💾 Guardando en disco local...")
processor.save_pretrained(LOCAL_MODEL_PATH)
model.save_pretrained(LOCAL_MODEL_PATH)

print(f"✅ Modelo guardado en: {LOCAL_MODEL_PATH}")