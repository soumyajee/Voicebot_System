import streamlit as st
import tempfile
import os
from pydub import AudioSegment
from gtts import gTTS
import assemblyai as aai
from faster_whisper import WhisperModel
import vosk
import json
import queue
import soundfile as sf

# ==============================
# CONFIGURATION
# ==============================
aai.settings.api_key = st.secrets.get("ASSEMBLYAI_API_KEY", "")

# ==============================
# SAVE AUDIO AS WAV (16kHz PCM)
# ==============================
def save_as_wav(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(audio_bytes)
            return tmp_wav.name
    except Exception as e:
        st.error(f"Failed to save audio: {e}")
        return None

# ==============================
# TRANSCRIBE FUNCTIONS
# ==============================
def transcribe_assembly(file_path):
    try:
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(language_code="en", punctuate=True, format_text=True)
        transcript = transcriber.transcribe(file_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"AssemblyAI error: {transcript.error}")
            return None

        return transcript.text if transcript.text else None
    except Exception as e:
        st.error(f"AssemblyAI SDK error: {e}")
        return None

def transcribe_faster_whisper(file_path):
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(file_path)
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        st.error(f"Faster-Whisper error: {e}")
        return None

def transcribe_vosk(file_path):
    try:
        model = vosk.Model("model")  # put vosk model in ./model directory
        q = queue.Queue()

        with sf.SoundFile(file_path) as audio_file:
            rec = vosk.KaldiRecognizer(model, audio_file.samplerate)
            while True:
                data = audio_file.buffer_read(4000, dtype="int16")
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    q.put(rec.Result())

        final_res = rec.FinalResult()
        return json.loads(final_res).get("text", "")
    except Exception as e:
        st.error(f"Vosk error: {e}")
        return None

def transcribe_audio(audio_bytes, engine="AssemblyAI"):
    file_path = save_as_wav(audio_bytes)
    if not file_path:
        return None

    if engine == "AssemblyAI":
        return transcribe_assembly(file_path)
    elif engine == "Faster-Whisper":
        return transcribe_faster_whisper(file_path)
    elif engine == "Vosk":
        return transcribe_vosk(file_path)
    return None

# ==============================
# TEXT-TO-SPEECH
# ==============================
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            tts.save(tmp_mp3.name)
            return tmp_mp3.name
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

# ==============================
# STREAMLIT UI
# ==============================
st.title("🎙️ Voice Bot with Multiple STT Engines")

engine = st.selectbox("Choose Transcription Engine", ["AssemblyAI", "Faster-Whisper", "Vosk"])

uploaded_audio = st.file_uploader("Upload your voice (WAV/MP3)", type=["wav", "mp3"])

if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav")
    audio_bytes = uploaded_audio.read()

    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_bytes, engine=engine)

        if transcription:
            st.success("✅ Transcription complete")
            st.write(f"**You said:** {transcription}")

            mp3_path = text_to_speech(f"You said: {transcription}")
            if mp3_path:
                st.audio(mp3_path, format="audio/mp3")
        else:
            st.error("❌ Could not transcribe audio")
