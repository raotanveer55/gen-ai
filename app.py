import streamlit as st
from transformers import pipeline

# Sidebar navigation
st.sidebar.title("🔍 NLP App Navigation")
section = st.sidebar.radio("Go to", ["Sentiment Analysis", "Text Classification", "Translation Analysis"])

# ====== Shared: Cached model loaders ======
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

@st.cache_resource
def load_translation_pipeline():
    return pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=-1)

# ====== Section 1: Sentiment Analysis ======
if section == "Sentiment Analysis":
    st.title("🧠 Sentiment Analysis")
    sentiment_pipeline = load_sentiment_pipeline()

    st.write("Enter English text to analyze sentiment:")
    input_text = st.text_area("Your text", "I love you\nI hate you", height=150)

    if st.button("Analyze Sentiment"):
        lines = [line for line in input_text.split('\n') if line.strip()]
        with st.spinner("Analyzing..."):
            results = sentiment_pipeline(lines)
        st.subheader("Results:")
        for i, res in enumerate(results):
            st.write(f"**Input:** {lines[i]}")
            st.write(f"→ Label: `{res['label']}` | Score: `{res['score']:.4f}`")
            st.markdown("---")

# ====== Section 2: Text Classification ======
elif section == "Text Classification":
    st.title("📂 Text Classification")
    st.write("This can be customized to use any classification model you like.")

    # Load the same model as placeholder
    classifier = load_sentiment_pipeline()

    user_text = st.text_input("Enter text to classify", "The service was excellent.")

    if st.button("Classify Text"):
        with st.spinner("Classifying..."):
            result = classifier(user_text)
        st.success("Classification Result:")
        st.write(result[0])

# ====== Section 3: Translation Analysis ======
elif section == "Translation Analysis":
    st.title("🌐 Translation Analysis")
    translator = load_translation_pipeline()

    st.write("Translate English text to French.")
    english_text = st.text_area("English Input", "I love working with AI.")

    if st.button("Translate"):
        with st.spinner("Translating..."):
            translation = translator(english_text)
        st.success("French Translation:")
        st.write(translation[0]['translation_text'])
