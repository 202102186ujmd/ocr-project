from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_path = "./models/paligemma-3b-pt-448"

processor = AutoProcessor.from_pretrained(model_path)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)

print("Modelo cargado correctamente ✅")