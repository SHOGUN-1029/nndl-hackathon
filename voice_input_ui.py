import streamlit as st
import whisper
import tempfile
import re
from transformers import pipeline

# ✅ Set page config
st.set_page_config(page_title="IPC Predictor from Audio", layout="centered")

# ✅ Load Whisper model (cached)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

model = load_whisper_model()

# ✅ Load summarization pipeline (downloads from Hugging Face)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ✅ Substitution dictionary for legal abstraction
substitutions = {
    "killed": "murdered", "murdered": "murdered", "girl": "woman",
    "beat up": "assaulted", "punched": "assaulted", "raped": "rape",
    "took bribe": "corruption", "bribed": "corruption",
    "drunk driving": "drunken driving", "burned": "arson",
    "set fire": "arson", "stole": "theft", "snatched": "robbery",
    "kidnapped": "kidnapping", "cheated": "cheating",
    "blackmailed": "criminal intimidation", "minor": "child"
}

legal_terms_set = set(substitutions.values())

# ✅ Legal keyword extractor (now applied on summary)
def extract_legal_keywords(text):
    text = text.lower()
    for informal, formal in substitutions.items():
        pattern = r'\b' + re.escape(informal) + r'\b'
        text = re.sub(pattern, formal, text)
    tokens = text.split()
    keywords = sorted(set(token for token in tokens if token in legal_terms_set))
    return keywords

# ✅ UI Section
st.title("🎙️ IPC Predictor from Regional Audio")
st.markdown("Upload an audio file in **MP3/WAV** format (regional or English). This app will:")
st.markdown("1. Transcribe and translate to English\n2. Summarize the incident\n3. Extract IPC-related legal keywords")

audio_file = st.file_uploader("Upload your audio file:", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file, format="audio/mp3")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        temp_audio_path = tmp.name

    with st.spinner("Transcribing and translating..."):
        result = model.transcribe(temp_audio_path, task="translate")
        transcript = result["text"]

    st.subheader("📝 Translated English Transcript")
    st.success(transcript)

    # ✅ Summarize transcript first
    with st.spinner("Summarizing incident..."):
        summary = summarizer(transcript, max_length=25, min_length=10, do_sample=False)[0]["summary_text"]

    st.subheader("🧠 Summary of the Incident")
    st.info(summary)

    # ✅ Extract legal keywords from summary
    keywords = extract_legal_keywords(summary)
    st.subheader("⚖️ Extracted Legal Keywords (from Summary)")
    if keywords:
        st.success(", ".join(keywords))
    else:
        st.warning("No legal terms found in the summary.")

else:
    st.info("Please upload a `.mp3` or `.wav` file to begin.")
