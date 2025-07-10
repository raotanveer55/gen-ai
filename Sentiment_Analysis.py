from transformers import pipeline

# Load sentiment analysis model
specific_model = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"  # ✅ default public sentiment model
)

# Sample data
data = ["I love you", "I hate you"]
result = specific_model(data)

print(result)