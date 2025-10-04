import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import requests
import json
import os
from gtts import gTTS
import tempfile

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
        self.frames.append(frame.to_ndarray())
        return frame

# ========================
# SESSION STATE
# ========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ========================
# TABS: Text Input vs Voice Input
# ========================
tab_text, tab_voice = st.tabs(["📝 Text Input", "🎤 Voice Input"])

# ------------------------
# TEXT INPUT TAB
# ------------------------
with tab_text:
    user_text = st.text_area("Type your message:", height=150, placeholder="Ask me anything...")

    if st.button("Send Text"):
        if user_text.strip():
            st.session_state["messages"].append({"role": "user", "content": user_text})
            
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
            else:
                st.error(f"API Error: {response.text}")

# ------------------------
# VOICE INPUT TAB
# ------------------------
with tab_voice:
    st.info("🎧 Speak into your microphone below")
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
            import numpy as np
            import soundfile as sf

            audio_np = np.concatenate(ctx.audio_processor.frames, axis=0)
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_wav.name, audio_np, 48000)  # sample rate may vary
            temp_wav.close()

            # Transcribe using SpeechRecognition
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_wav.name) as source:
                audio_data = recognizer.record(source)
                try:
                    user_input = recognizer.recognize_google(audio_data)
                    st.markdown(f"**🧑 You said:** {user_input}")
                    st.session_state["messages"].append({"role": "user", "content": user_input})

                    # Call OpenRouter API
                    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                    data = {"model": MODEL, "messages": st.session_state["messages"]}
                    response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(data))
                    if response.status_code == 200:
                        bot_reply = response.json()["choices"][0]["message"]["content"]
                        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
                        st.markdown(f"**🤖 Bot:** {bot_reply}")

                        # Optional: Text-to-speech
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

# ------------------------
# Display chat history
# ------------------------
st.markdown("---")
st.subheader("💬 Chat History")
for msg in st.session_state["messages"]:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")
