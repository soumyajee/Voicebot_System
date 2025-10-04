import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os
import speech_recognition as sr
from dotenv import load_dotenv
from st_audiorec import st_audiorec  # ✅ browser-based recorder

# -----------------------------
# Load API key from .env file
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice Bot")
st.write("Chat by recording your voice or typing below.")

# -----------------------------
# Helper functions
# -----------------------------
def get_response(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None

def text_to_speech(text):
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Record your voice (browser mic)")

    wav_audio_data = st_audiorec()  # ✅ recorder widget

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format="audio/wav")
        
        # Save audio to temp file for SpeechRecognition
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_file.name, "wb") as f:
            f.write(wav_audio_data)
        
        if st.button("📝 Transcribe & Get Response", type="primary"):
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file.name) as source:
                audio_data = recognizer.record(source)
                try:
                    user_input = recognizer.recognize_google(audio_data, language="en-US")
                    st.markdown("### 🎤 You said:")
                    st.info(f"**\"{user_input}\"**")

                    with st.spinner("🤔 Getting response..."):
                        answer = get_response(user_input)

                    if answer:
                        st.markdown("---")
                        st.markdown("### 🤖 Bot Response:")
                        st.write(answer)

                        # Generate audio response
                        audio_file = text_to_speech(answer)
                        if audio_file:
                            st.audio(audio_file, format="audio/mp3")
                            with open(audio_file, "rb") as file:
                                st.download_button(
                                    label="📥 Download",
                                    data=file,
                                    file_name="response.mp3",
                                    mime="audio/mp3"
                                )
                            os.unlink(audio_file)
                except sr.UnknownValueError:
                    st.error("❌ Could not understand the audio")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

with tab2:
    st.subheader("Type your message")
    user_text = st.text_area("Message:", height=150)
    if st.button("Send Text", type="primary"):
        if user_text:
            with st.spinner("🤔 Thinking..."):
                answer = get_response(user_text)
            if answer:
                st.markdown("### 🤖 Bot Response:")
                st.write(answer)

                audio_file = text_to_speech(answer)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                    with open(audio_file, "rb") as file:
                        st.download_button(
                            label="📥 Download",
                            data=file,
                            file_name="response.mp3",
                            mime="audio/mp3"
                        )
                    os.unlink(audio_file)

st.markdown("---")
st.caption("💡 Tip: Browser mic recording works on Render without system libraries.")
