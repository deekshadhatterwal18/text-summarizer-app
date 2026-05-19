import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("deekshadhatterwal18/text-summarizer-t5")
    tokenizer = T5Tokenizer.from_pretrained("deekshadhatterwal18/text-summarizer-t5")
    return model, tokenizer

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue, model, tokenizer, device):
    dialogue = clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

st.title("📝 Text Summarizer")
st.write("Paste your text below to get a summary!")

with st.spinner("Loading model... (first time takes a while)"):
    model, tokenizer = load_model()
    device = get_device()

st.success("Model ready! ✅")

input_text = st.text_area("Enter text here:", height=200, placeholder="Paste your text here...")

if st.button("Summarize "):
    if input_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_dialogue(input_text, model, tokenizer, device)
        st.subheader("Summary:")
        st.success(summary)