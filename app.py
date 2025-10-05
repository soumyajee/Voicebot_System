import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import wave
from groq import Groq

# ------------------------------
# Load API keys
# ------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq client for Whisper
groq_client = Groq(api_key=GROQ_API_KEY)

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")
st.write("Record voice directly in your browser! 🎧 (Grant mic permission when prompted.)")

# ------------------------------
# Microphone Setup Guide
# ------------------------------
with st.expander("📱 Microphone Setup (Important!)"):
    st.markdown("""
    ### Before Recording:
    1. **Allow** microphone access in your browser (popup will appear on first record).
    2. **Test** your mic in browser settings.
    3. **Use headphones** if in a noisy environment.
    4. Recording happens in-browser—no server mic needed.
    5. **Click 'Start Recording'** to begin, then **'Stop'** to process.
    6. Keep recordings short (5-30s) for best transcription results.
    """)

# ------------------------------
# Helper Functions
# ------------------------------

# Get LLM response from OpenRouter
def get_response(prompt):
    if not OPENROUTER_API_KEY:
        st.error("API key not found. Add OPENROUTER_API_KEY to .env file.")
        return None
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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

# Convert text to speech
def text_to_speech(text):
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

# Save audio bytes as WAV
def save_as_wav(audio_bytes, sample_rate=16000, channels=1, sample_width=2):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            return tmp_file.name
    except Exception as e:
        st.error(f"❌ Error saving WAV file: {e}")
        return None

# Transcribe audio using Groq Whisper
def transcribe_audio(audio_bytes):
    if not audio_bytes:
        return None
    try:
        tmp_file_path = save_as_wav(audio_bytes)
        if not tmp_file_path:
            return None

        # Send file to Groq Whisper
        with open(tmp_file_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3"
            )

        os.unlink(tmp_file_path)
        return transcription.text
    except Exception as e:
        st.error(f"❌ Whisper Transcription Error: {e}")
        return None

# ------------------------------
# Main Interface Tabs
# ------------------------------
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

# ------------------------------
# Voice Input Tab
# ------------------------------
with tab1:
    st.subheader("Browser-Based Voice Recording")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        audio_data = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="🛑 Stop Recording",
            key="mic_recorder"
        )
    
    with col2:
        if st.button("🔄 Clear Recording", use_container_width=True):
            st.session_state.audio_bytes = None
            st.rerun()
    
    if audio_data:
        try:
            audio_bytes = audio_data.get('bytes', audio_data.get('data', None))
            if not audio_bytes:
                st.error("❌ No audio data found.")
            else:
                st.session_state.audio_bytes = audio_bytes
                st.success("✅ Recording complete!")
                
                st.audio(audio_bytes, format="audio/wav")
                
                if st.button("📝 Transcribe & Get Response", type="primary", use_container_width=True):
                    with st.spinner("🔄 Transcribing with Whisper (Groq)..."):
                        user_input = transcribe_audio(audio_bytes)
                        
                        if user_input:
                            st.markdown("### 🎤 You said:")
                            st.info(f'"{user_input}"')
                            
                            with st.spinner("🤔 Getting AI response..."):
                                answer = get_response(user_input)
                            
                            if answer:
                                st.markdown("---")
                                st.markdown("### 🤖 Bot Response:")
                                st.write(answer)
                                
                                with st.spinner("🔊 Generating voice response..."):
                                    tts_file = text_to_speech(answer)
                                
                                if tts_file:
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.audio(tts_file, format="audio/mp3")
                                    with col_b:
                                        with open(tts_file, "rb") as f:
                                            st.download_button(
                                                label="📥 Download",
                                                data=f,
                                                file_name="response.mp3",
                                                mime="audio/mp3",
                                                key="voice_download",
                                                use_container_width=True
                                            )
                                    os.unlink(tts_file)
        except AttributeError:
            st.error("❌ Invalid audio data format. Please try again.")

# ------------------------------
# Text Input Tab
# ------------------------------
with tab2:
    st.subheader("Text Input")
    user_text = st.text_area("Type your message:", height=150, placeholder="Ask me anything...")
    
    if st.button("Send Text", type="primary"):
        if user_text.strip():
            with st.spinner("🤔 Thinking..."):
                answer = get_response(user_text)
            
            if answer:
                st.markdown("### 🤖 Bot Response:")
                st.write(answer)
                
                with st.spinner("🔊 Generating voice..."):
                    tts_file = text_to_speech(answer)
                
                if tts_file:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.audio(tts_file, format="audio/mp3")
                    with col_b:
                        with open(tts_file, "rb") as f:
                            st.download_button(
                                label="📥 Download",
                                data=f,
                                file_name="response.mp3",
                                mime="audio/mp3",
                                key="text_download",
                                use_container_width=True
                            )
                    os.unlink(tts_file)
        else:
            st.warning("Please enter some text first.")

# ------------------------------
# Troubleshooting Section
# ------------------------------
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### If recording doesn't work:
    - **Browser Permissions**: Ensure mic access is granted.
    - **HTTPS Required**: Use HTTPS (Streamlit Cloud enforces this).
    - **No Audio Detected**: Speak clearly; test mic in another app/site.
    - **Groq Transcription Fails**:
        - Ensure `GROQ_API_KEY` is set in `.env`.
        - Keep recordings short (5-30s).
        - Check if audio file size is non-zero.
    - **Component Issues**: Ensure `streamlit-mic-recorder==0.0.8` is installed.
    """)

st.markdown("---")
st.caption("Powered by Groq Whisper v3 & OpenRouter GPT")
