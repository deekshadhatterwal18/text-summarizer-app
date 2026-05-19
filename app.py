import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="centered"
)

# LOAD MODEL - cached so loads only once
@st.cache_resource(show_spinner=False)
def load_model():
    model = T5ForConditionalGeneration.from_pretrained(
        "deekshadhatterwal18/text-summarizer-t5"
    )
    tokenizer = T5Tokenizer.from_pretrained(
        "deekshadhatterwal18/text-summarizer-t5"
    )
    model.eval()  # inference mode - saves memory
    return model, tokenizer

# DEVICE - cpu only for deployment (streamlit cloud has no GPU)
@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cpu")

# CLEAN TEXT
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

# PDF TEXT - lazy import to reduce startup load
def extract_pdf_text(pdf_file):
    try:
        import PyPDF2
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text.strip()
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return ""

# TXT TEXT
def extract_txt_text(txt_file):
    return txt_file.read().decode("utf-8")

# SUMMARIZE - torch.no_grad() saves memory during inference
def summarize_text(text, model, tokenizer, device, max_len):
    text = clean_data(text)

    # limit input length to avoid heavy processing
    words = text.split()
    if len(words) > 400:
        text = " ".join(words[:400])

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():  # no gradient = less memory
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=2,       # reduced from 4 to 2 - faster, less memory
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.title("📝 AI Text Summarizer")
st.write("Upload a PDF/TXT file or paste text to get a summary.")

# LOAD MODEL
with st.spinner("Loading model... (first time only)"):
    model, tokenizer = load_model()
    device = get_device()

st.success("Model ready ✅")

# SETTINGS
col1, col2 = st.columns(2)

with col1:
    summary_length = st.selectbox(
        "Summary Length",
        ["Short (50 words)", "Medium (100 words)", "Detailed (150 words)"]
    )

with col2:
    summary_style = st.radio("Summary Style", ["Paragraph", "Bullet Points"])

# MAX LENGTH
length_map = {
    "Short (50 words)": 50,
    "Medium (100 words)": 100,
    "Detailed (150 words)": 150
}
max_len = length_map[summary_length]
bullet_mode = summary_style == "Bullet Points"

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

# TEXT INPUT
input_text = st.text_area(
    "Or paste text here:",
    height=200,
    placeholder="Paste your dialogue or text here..."
)

# SUMMARIZE BUTTON
if st.button("Summarize 🚀", use_container_width=True):

    final_text = ""

    if uploaded_file is not None:
        with st.spinner("Reading file..."):
            if uploaded_file.type == "application/pdf":
                final_text = extract_pdf_text(uploaded_file)
            else:
                final_text = extract_txt_text(uploaded_file)

    elif input_text.strip():
        final_text = input_text

    else:
        st.warning("Please upload a file or enter some text!")
        st.stop()

    with st.spinner("Generating summary..."):
        if bullet_mode:
            final_text = "summarize: " + final_text
        summary = summarize_text(final_text, model, tokenizer, device, max_len)

    st.subheader("Summary")
    st.success(summary)

    # COPY + DOWNLOAD
    st.text_area("Copy Summary", summary, height=120)
    st.download_button(
        label="⬇️ Download Summary",
        data=summary,
        file_name="summary.txt",
        mime="text/plain"
    )

    # WORD STATS
    original_words = len(final_text.split())
    summary_words = len(summary.split())
    compression = round((summary_words / original_words) * 100, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Words", original_words)
    col2.metric("Summary Words", summary_words)
    col3.metric("Compression", f"{compression}%")

    # SAVE TO HISTORY (max 5 items only)
    st.session_state.history = st.session_state.history[-4:]
    st.session_state.history.append({
        "input": final_text[:100],
        "summary": summary
    })

# HISTORY
if st.session_state.history:
    with st.expander("📋 Recent Summaries"):
        for item in reversed(st.session_state.history):
            st.markdown(f"**Input:** {item['input']}...")
            st.success(item["summary"])
            st.divider()
