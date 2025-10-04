import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import requests
import json
import tempfile
import os
import numpy as np
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS

# ========================
# CONFIG
# ========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "gpt-4o-mini"

st.title("🎤 AI Voice & Text Bot")

# ========================
# AUDIO CAPTURE CLASS
# ========================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

# ========================
# SESSION STATE
# ========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "transcriptions" not in st.session_state:
    st.session_state["transcriptions"] = []  # ✅ store all speech-to-text here

# ========================
# VOICE INPUT TAB
# ========================
st.subheader("🎤 Voice Input (WebRTC)")
ctx = webrtc_streamer(
    key="voice-bot",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

if st.button("Transcribe & Send"):
    if ctx.audio_processor and ctx.audio_processor.frames:
        # Save temporary WAV
        audio_np = np.concatenate(ctx.audio_processor.frames, axis=0)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, audio_np, 48000)
        temp_wav.close()

        # Transcribe using SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio_data)
                st.markdown(f"**🧑 You said:** {user_input}")

                # Save transcription
                st.session_state.transcriptions.append(user_input)

                # Store user input in chat history
                st.session_state["messages"].append({"role": "user", "content": user_input})

                # Call OpenRouter API
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {"model": MODEL, "messages": st.session_state["messages"]}
                response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(data))
                if response.status_code == 200:
                    bot_reply = response.json()["choices"][0]["message"]["content"]
                    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
                    st.markdown(f"**🤖 Bot:** {bot_reply}")

                    # Optional: TTS reply
                    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts = gTTS(bot_reply)
                    tts.save(tts_file.name)
                    st.audio(tts_file.name)
                    tts_file.close()
                else:
                    st.error(f"API Error: {response.text}")

            except sr.UnknownValueError:
                st.error("❌ Could not understand the audio.")
        os.unlink(temp_wav.name)

# ========================
# TEXT INPUT TAB
# ========================
st.subheader("📝 Text Input")
user_text = st.text_input("Type your question:")
if st.button("Send Text"):
    if user_text.strip():
        st.session_state["messages"].append({"role": "user", "content": user_text})
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        data = {"model": MODEL, "messages": st.session_state["messages"]}
        response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            bot_reply = response.json()["choices"][0]["message"]["content"]
            st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
            st.markdown(f"**🤖 Bot:** {bot_reply}")
        else:
            st.error(f"API Error: {response.text}")

# ========================
# DISPLAY CHAT HISTORY
# ========================
st.markdown("---")
st.subheader("💬 Chat History")
for msg in st.session_state["messages"]:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# ========================
# DISPLAY ALL TRANSCRIPTIONS
# ========================
if st.session_state["transcriptions"]:
    st.markdown("---")
    st.subheader("📝 Speech-to-Text Transcriptions")
    for idx, text in enumerate(st.session_state["transcriptions"], 1):
        st.markdown(f"{idx}. {text}")
