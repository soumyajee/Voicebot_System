import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os
import sounddevice as sd   # ✅ replaces pyaudio
import numpy as np
import wave
import speech_recognition as sr
from dotenv import load_dotenv  # ✅ for .env support

# -----------------------------
# Load API key from .env file
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")  # ✅ safely loaded from .env

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")
st.write("Record from any connected microphone! 🎧")

# Microphone setup guide
with st.expander("📱 Microphone Setup (Important!)"):
    st.markdown("""
    ### Before Recording:
    1. **Connect** your microphone (built-in, USB, Bluetooth, etc.) to this device.
    2. **Select** your microphone from the dropdown below.
    3. **Test** your microphone in system settings to ensure it’s working.
    4. **Refresh** this page after connecting a new device.
    """)

# Initialize session state
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

# -----------------------------
# OpenRouter API
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
    except requests.RequestException as e:
        st.error(f"❌ API Error: {e}")
        return None

# -----------------------------
# Text to Speech
# -----------------------------
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
# Record audio with sounddevice
# -----------------------------
def record_audio(duration=5, samplerate=16000):
    try:
        st.info(f"🎤 Recording for {duration} seconds...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # wait until done

        # Save to WAV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
        wf.close()

        return temp_file.name
    except Exception as e:
        st.error(f"❌ Recording Error: {e}")
        return None

# -----------------------------
# Main interface tabs
# -----------------------------
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Live Recording from Microphone")
    record_duration = st.selectbox("Recording Duration (seconds)", [3, 5, 7, 10], index=1)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            with st.spinner(f"🔴 Recording for {record_duration} seconds... Speak now!"):
                audio_file = record_audio(duration=record_duration)
                if audio_file:
                    st.session_state.audio_file = audio_file
                    st.success("✅ Recording complete!")
                    st.rerun()

    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.audio_file = None
            st.rerun()

    # Process recorded audio
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        st.write("---")
        st.success("✅ Audio recorded successfully!")

        # Play recorded audio
        with open(st.session_state.audio_file, "rb") as audio:
            st.audio(audio.read(), format="audio/wav")

        # Transcribe button
        if st.button("📝 Transcribe & Get Response", type="primary"):
            with st.spinner("🔄 Transcribing your speech..."):
                try:
                    recognizer = sr.Recognizer()
                    recognizer.energy_threshold = 300
                    recognizer.dynamic_energy_threshold = True

                    with sr.AudioFile(st.session_state.audio_file) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data = recognizer.record(source)

                        user_input = recognizer.recognize_google(audio_data, language="en-US")

                        st.markdown("### 🎤 You said:")
                        st.info(f"**\"{user_input}\"**")

                        # Get bot response
                        with st.spinner("🤔 Getting response..."):
                            answer = get_response(user_input)

                        if answer:
                            st.markdown("---")
                            st.markdown("### 🤖 Bot Response:")
                            st.write(answer)

                            # Generate audio response
                            audio_file = text_to_speech(answer)
                            if audio_file:
                                st.success("🔊 Audio response:")
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.audio(audio_file, format="audio/mp3")
                                with col_b:
                                    with open(audio_file, "rb") as file:
                                        st.download_button(
                                            label="📥 Download",
                                            data=file,
                                            file_name="response.mp3",
                                            mime="audio/mp3",
                                            use_container_width=True
                                        )
                                os.unlink(audio_file)

                except sr.UnknownValueError:
                    st.error("❌ Could not understand the audio.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

with tab2:
    st.subheader("Text Input")
    user_text = st.text_area("Type your message:", height=150, placeholder="Ask me anything...")

    if st.button("Send Text", type="primary"):
        if user_text:
            with st.spinner("🤔 Thinking..."):
                answer = get_response(user_text)

            if answer:
                st.markdown("### 🤖 Bot Response:")
                st.write(answer)

                # Generate audio
                audio_file = text_to_speech(answer)
                if audio_file:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.audio(audio_file, format="audio/mp3")
                    with col_b:
                        with open(audio_file, "rb") as file:
                            st.download_button(
                                label="📥 Download",
                                data=file,
                                file_name="response.mp3",
                                mime="audio/mp3",
                                use_container_width=True
                            )
                    os.unlink(audio_file)

st.markdown("---")
st.caption("💡 Tip: Use your microphone to ask questions and hear spoken answers!")

