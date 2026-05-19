import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import PyPDF2
import streamlit.components.v1 as components

# Setup page
st.set_page_config(
    page_title="SnapText",
    page_icon="⚡",
    layout="centered"
)

# Custom UI
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

h1 {
    color: #c084fc !important;
    text-align: center;
    font-size: 3rem !important;
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    padding: 10px;
}

.stDownloadButton > button {
    width: 100%;
    border-radius: 10px;
}

.stTextArea textarea {
    background-color: #1e1b4b !important;
    color: white !important;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Load transformer model
@st.cache_resource
def load_model():

    model = T5ForConditionalGeneration.from_pretrained(
        "deekshadhatterwal18/text-summarizer-t5"
    )

    tokenizer = T5Tokenizer.from_pretrained(
        "deekshadhatterwal18/text-summarizer-t5"
    )

    model.eval()

    return model, tokenizer

# Detect device
def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

# Clean input text
def clean_data(text):

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    return text.strip().lower()

# Extract PDF text
def extract_pdf_text(pdf_file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page in pdf_reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + " "

    return text

# Extract TXT text
def extract_txt_text(txt_file):

    return txt_file.read().decode("utf-8")

# Generate AI summary
def summarize_text(
    text,
    model,
    tokenizer,
    device,
    max_len
):

    text = clean_data(text)

    words = text.split()

    if len(words) > 800:
        text = " ".join(words[:800])

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    model.to(device)

    with torch.no_grad():

        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            min_length=max_len // 2,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

# Store history
if "history" not in st.session_state:
    st.session_state.history = []

# Load model
with st.spinner("Loading Model..."):

    model, tokenizer = load_model()

    device = get_device()

# Title
st.title("⚡ SnapText")

st.divider()

# Settings
col1, col2 = st.columns(2)

with col1:

    summary_length = st.selectbox(
        "Summary Length",
        [
            "Short",
            "Medium",
            "Detailed"
        ]
    )

with col2:

    summary_style = st.radio(
        "Summary Style",
        [
            "Paragraph",
            "Bullet Points"
        ]
    )

# Length values
length_map = {
    "Short": 50,
    "Medium": 120,
    "Detailed": 220
}

max_len = length_map[summary_length]

bullet_mode = summary_style == "Bullet Points"

st.divider()

# Upload files
uploaded_file = st.file_uploader(
    "Upload PDF or TXT File",
    type=["pdf", "txt"]
)

# Input text
input_text = st.text_area(
    "Paste Dialogue or Text",
    height=250,
    placeholder="Paste your conversation or text here..."
)

# Generate button
if st.button("✨ Generate Summary"):

    final_text = ""

    # Read uploaded file
    if uploaded_file is not None:

        with st.spinner("Reading file..."):

            if uploaded_file.type == "application/pdf":

                final_text = extract_pdf_text(uploaded_file)

            else:

                final_text = extract_txt_text(uploaded_file)

    elif input_text.strip() != "":

        final_text = input_text

    else:

        st.warning("Please upload file or enter text!")

        st.stop()

    input_for_model = "summarize: " + final_text

    # Generate summary
    with st.spinner("Generating Summary..."):

        summary = summarize_text(
            input_for_model,
            model,
            tokenizer,
            device,
            max_len
        )

    st.divider()

    # Show summary
    st.subheader("📋 Summary")

    if bullet_mode:

        bullet_summary = summary.split(". ")

        for point in bullet_summary:

            if point.strip() != "":
                st.markdown(f"- {point}")

    else:

        st.success(summary)

   
    # Download summary
    st.download_button(
        label="⬇️ Download Summary",
        data=summary,
        file_name="summary.txt",
        mime="text/plain"
    )

    st.divider()

    # Word statistics
    original_words = len(final_text.split())

    summary_words = len(summary.split())

    compression = round(
        ((original_words - summary_words) / original_words) * 100,
        2
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original Words",
        original_words
    )

    c2.metric(
        "Summary Words",
        summary_words
    )

    c3.metric(
        "Compression",
        f"{compression}%"
    )

    # Save history
    st.session_state.history.append({

        "input": final_text[:100],

        "summary": summary
    })

# Show history
if st.session_state.history:

    st.divider()

    with st.expander("🕘 Recent Summaries"):

        for item in reversed(
            st.session_state.history
        ):

            st.markdown("### Input")

            st.write(
                item["input"] + "..."
            )

            st.markdown("### Summary")

            st.success(
                item["summary"]
            )

            st.divider()
