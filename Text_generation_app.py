import streamlit as st
from transformers import pipeline
import re

# Page setup
st.set_page_config(page_title="Text Generator + QA", layout="centered")
st.title("🤖 GPT2 / DistilGPT2 - Text Generator & QA")

# Sidebar model choice
model_choice = st.sidebar.selectbox("Choose a model:", ["gpt2", "distilgpt2"])

# Load model
@st.cache_resource
def load_model(name):
    return pipeline("text-generation", model=name)

text_generator = load_model(model_choice)

# Main input
user_input = st.text_area("Enter a prompt or ask a question:", "What is the capital of France?")

# Question toggle
is_question = st.checkbox("This is a question (return short answer)", value=True)

# Sliders
st.sidebar.subheader("⚙️ Generation Settings")
max_len = st.sidebar.slider("Max tokens", 20, 512, 150)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
top_k = st.sidebar.slider("Top-K", 10, 100, 50)
top_p = st.sidebar.slider("Top-P", 0.1, 1.0, 0.95)
num_outputs = st.sidebar.slider("Number of outputs", 1, 5, 1)

# Generate response
if st.button("Generate Response"):
    if not user_input.strip():
        st.warning("Please enter a prompt or question.")
    else:
        with st.spinner("Generating..."):
            # QA prompt formatting
            if is_question:
                prompt = f"Q: {user_input.strip()}\nA:"
            else:
                prompt = user_input.strip()

            results = text_generator(
                prompt,
                max_length=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_outputs,
                no_repeat_ngram_size=3,
                pad_token_id=50256  # EOS token for GPT2
            )

            st.subheader("🧾 Generated Output:")
            for i, res in enumerate(results):
                text = res["generated_text"]
                if is_question and "A:" in text:
                    text = text.split("A:")[1].strip()
                    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
                    text = text.split("\n")[0].split(".")[0] + "."
                st.markdown(f"**Output {i+1}:**")
                st.write(text)
