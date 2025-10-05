import streamlit as st
import requests
from gtts import gTTS
import os
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import wave
from groq import Groq
from time import sleep
import time

# ------------------------------
# Load API keys
# ------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Local directory for saving audio files
AUDIO_DIR = "./audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)  # Create directory if it doesn't exist

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")
st.write("Record voice directly in your browser! 🎧 (Grant mic permission when prompted.)")

# Microphone setup guide
with st.expander("📱 Microphone Setup (Important!)"):
    st.markdown("""
    ### Before Recording:
    1. **Allow** microphone access in your browser (popup will appear on first record).
    2. **Test** your mic in browser settings (e.g., Chrome: chrome://settings/content/microphone).
    3. **Use headphones** if in a noisy environment.
    4. Recording happens in-browser—no server mic needed.
    5. **Click 'Start Recording'** to begin, then **'Stop'** to process.
    6. Keep recordings short (5-30s) for best transcription results.
    """)

# Initialize session state for audio
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

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
        st.error(f"❌ OpenRouter API Error: {e}")
        return None

# Convert text to speech
def text_to_speech(text):
    try:
        tts = gTTS(text)
        temp_file = os.path.join(AUDIO_DIR, f"tts_{int(time.time())}.mp3")
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

# Save audio bytes as WAV to local directory
def save_as_wav(audio_bytes, sample_rate=44100, channels=1, sample_width=2):
    try:
        # Generate unique filename with timestamp
        filename = f"recording_{int(time.time())}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        st.write(f"Debug: Saved WAV file at {file_path}")
        return file_path
    except Exception as e:
        st.error(f"❌ Error saving WAV file: {e}")
        return None

# Transcribe audio using Groq Whisper-large-v3
def transcribe_audio(audio_bytes):
    if not audio_bytes:
        st.error("❌ No audio data provided.")
        return None
    if not groq_client:
        st.error("❌ Groq API key not found. Add GROQ_API_KEY to .env file.")
        return None
    try:
        file_path = save_as_wav(audio_bytes, sample_rate=44100, channels=1, sample_width=2)
        if not file_path:
            return None

        file_size = os.path.getsize(file_path)
        st.write(f"Debug: Saved WAV file at {file_path}, size: {file_size} bytes")
        
        # Offer to download WAV for debugging
        with open(file_path, "rb") as f:
            st.download_button("Download WAV for Debug", f, file_name=os.path.basename(file_path), mime="audio/wav")

        if file_size < 100:
            st.error("❌ Audio file is too small or empty. Try recording again.")
            os.unlink(file_path)
            return None

        retries = 3
        for attempt in range(retries):
            try:
                with open(file_path, "rb") as audio_file:
                    transcript = groq_client.audio.transcriptions.create(
                        model="whisper-large-v3",  # Try "whisper-large-v3-turbo" if issues persist
                        file=audio_file,
                        response_format="text",
                        language="en"
                    )
                os.unlink(file_path)
                st.write(f"Debug: Whisper transcription successful: {transcript}")
                return transcript
            except Exception as e:
                if attempt < retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{retries})... Error: {str(e)}")
                    sleep(2)
                else:
                    raise e
    except Exception as e:
        if "500" in str(e):
            st.error("❌ Groq Server Error: Internal server issue. Try again later or use 'whisper-large-v3-turbo'.")
        else:
            st.error(f"❌ Groq Whisper Error: {e}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        return None

# ------------------------------
# Main Interface Tabs
# ------------------------------
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

# ------------------------------
# Voice Input Tab
# ----------------------
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
                    with st.spinner("🔄 Transcribing with Groq Whisper..."):
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
            st.error("❌ Invalid audio data format. Please try recording again.")

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
    - **Browser Permissions**: Ensure mic access is granted (check site settings).
    - **HTTPS Required**: Use HTTPS (Render enforces this).
    - **No Audio Detected**: Speak clearly; test mic in another app/site.
    - **Groq Transcription Fails**:
        - Ensure `GROQ_API_KEY` is set in `.env` and Render's environment variables.
        - Keep recordings short (5-30s, <25MB).
        - Download and test WAV file locally to ensure it's not corrupted.
        - For 500 errors, try model `whisper-large-v3-turbo` or retry later.
        - Test API key with `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`.
    - **Component Issues**: Ensure `streamlit-mic-recorder==0.0.8` is installed.
    - **Local Storage**: Ensure `./audio_files/` is writable (check permissions on Render).
    - **Render Deployment**: Check logs for dependency or network errors.
    """)

st.markdown("---")
st.caption("Powered by Groq Whisper-large-v3 & OpenRouter GPT")
