from transformers import pipeline

# Английский → Французский
translator = pipeline("translation_en_to_ru")  # по умолчанию использует t5-base
result = translator("How old are you?")