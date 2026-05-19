import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    h1 {
        color: #c084fc !important;
        text-align: center;
        font-size: 2.5rem !important;
        margin-bottom: 0.2rem !important;
    }
    .subtitle {
        text-align: center;
        color: #a78bfa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border: 1px solid #7c3aed !important;
        border-radius: 12px !important;
    }
    .stSelectbox > div > div {
        background-color: #1e1b4b !important;
        color: #e2e8f0 !important;
        border: 1px solid #7c3aed !important;
        border-radius: 10px !important;
    }
    .stRadio label { color: #c084fc !important; }
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #6d28d9, #9333ea) !important;
        transform: scale(1.02);
    }
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid #7c3aed !important;
        color: #c084fc !important;
        border-radius: 10px !important;
        width: 100% !important;
    }
    .stSuccess {
        background-color: #1e1b4b !important;
        border-left: 4px solid #a855f7 !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }
    [data-testid="stMetric"] {
        background-color: #1e1b4b;
        border: 1px solid #7c3aed;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    [data-testid="stMetricLabel"] { color: #a78bfa !important; }
    [data-testid="stMetricValue"] { color: #c084fc !important; }
    [data-testid="stFileUploader"] {
        background-color: #1e1b4b !important;
        border: 1px dashed #7c3aed !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    .streamlit-expanderHeader {
        background-color: #1e1b4b !important;
        color: #c084fc !important;
        border-radius: 10px !important;
    }
    label, .stRadio > label { color: #a78bfa !important; font-weight: 500 !important; }
    hr { border-color: #4c1d95 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("deekshadhatterwal18/text-summarizer-t5")
    tokenizer = T5Tokenizer.from_pretrained("deekshadhatterwal18/text-summarizer-t5")
    model.eval()
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cpu")

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

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

def extract_txt_text(txt_file):
    return txt_file.read().decode("utf-8")

def summarize_text(text, model, tokenizer, device, max_len):
    text = clean_data(text)
    words = text.split()
    if len(words) > 400:
        text = " ".join(words[:400])
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=2,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if "history" not in st.session_state:
    st.session_state.history = []

model, tokenizer = load_model()
device = get_device()

st.title("📝 AI Text Summarizer")
st.divider()

col1, col2 = st.columns(2)
with col1:
    summary_length = st.selectbox("Summary Length", ["Short (50 words)", "Medium (100 words)", "Detailed (150 words)"])
with col2:
    summary_style = st.radio("Summary Style", ["Paragraph", "Bullet Points"])

length_map = {"Short (50 words)": 50, "Medium (100 words)": 100, "Detailed (150 words)": 150}
max_len = length_map[summary_length]
bullet_mode = summary_style == "Bullet Points"

st.divider()

uploaded_file = st.file_uploader("📂 Upload PDF or TXT", type=["pdf", "txt"])
input_text = st.text_area("✏️ Or paste text here:", height=200, placeholder="Paste your dialogue or text here...")

if st.button("Summarize"):
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
        st.warning("⚠️ Please upload a file or enter some text!")
        st.stop()

    with st.spinner("Generating summary..."):
        input_for_model = ("summarize: " + final_text) if bullet_mode else final_text
        summary = summarize_text(input_for_model, model, tokenizer, device, max_len)

    st.divider()
    st.subheader("📋 Summary")
    st.success(summary)

    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Copy Summary", summary, height=120)
    with col2:
        st.download_button(label="⬇️ Download Summary", data=summary, file_name="summary.txt", mime="text/plain")

    st.divider()
    original_words = len(final_text.split())
    summary_words = len(summary.split())
    compression = round((summary_words / original_words) * 100, 2)
    c1, c2, c3 = st.columns(3)
    c1.metric(" Original Words", original_words)
    c2.metric("Summary Words", summary_words)
    c3.metric("Compression", f"{compression}%")

    st.session_state.history = st.session_state.history[-4:]
    st.session_state.history.append({"input": final_text[:100], "summary": summary})

if st.session_state.history:
    st.divider()
    with st.expander("🕘 Recent Summaries"):
        for item in reversed(st.session_state.history):
            st.markdown(f"**Input:** {item['input']}...")
            st.success(item["summary"])
            st.divider()
