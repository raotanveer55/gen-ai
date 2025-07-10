from transformers import pipeline

# Load German to English translation model
specific_model = pipeline(
    task="translation_de_to_en",
    model="Helsinki-NLP/opus-mt-de-en"
)

# Sample data in German
data = ["Ich liebe dich", "Ich hasse dich"]
result = specific_model(data)

print(result)