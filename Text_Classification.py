from transformers import pipeline

# Load the emotion classification model
specific_model = pipeline(
    task="text-classification",
    model="bhadresh-savani/bert-base-uncased-emotion"
)

data = ["I love you", "I hate you"]
result = specific_model(data)
print(result)
