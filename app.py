import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os
import io
import speech_recognition as sr
from dotenv import load_dotenv

# Try to import audio_recorder_streamlit, fall back to file uploader if not available
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    st.warning("⚠️ Audio recorder module not available. Using file upload instead.")

# -----------------------------
# Load API key from .env file
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")
st.write("Record or upload audio to interact! 🎧")

# Microphone setup guide
with st.expander("📱 Audio Input Setup (Important!)"):
    st.markdown("""
    ### Two Ways to Use:
    1. **Live Recording** (if available): Click record button and allow microphone access.
    2. **File Upload**: Upload a pre-recorded audio file (WAV, MP3, M4A).
    
    ### Tips:
    - Use Chrome, Firefox, or Edge for best compatibility.
    - Speak clearly into your device's mic.
    - For file upload: Keep recordings under 1 minute for best results.
    - Check browser permissions if recording doesn't work.
    """)

# Initialize session state
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

# Function to get response from OpenRouter
def get_response(prompt):
    if not API_KEY:
        st.error("❌ API key not found. Please set OPENROUTER_API_KEY in your .env file.")
        return None
    
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

# Function to convert text to speech
def text_to_speech(text):
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

# Function to transcribe audio from bytes
def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        # Save bytes to temporary WAV file for SpeechRecognition
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        with sr.AudioFile(tmp_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        return text
        
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio. Please speak clearly and reduce noise.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

# Function to process audio (used by both recorder and uploader)
def process_audio(audio_bytes):
    if not audio_bytes:
        return
    
    st.success("✅ Audio received!")
    st.audio(audio_bytes, format="audio/wav")
    
    # Transcribe and get response
    with st.spinner("🔄 Transcribing your speech..."):
        user_input = transcribe_audio(audio_bytes)
        if user_input:
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

# Main interface tabs
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Audio Input")
    
    if AUDIO_RECORDER_AVAILABLE:
        # Use audio recorder if available
        st.write("**Live Recording from Browser Microphone**")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8b923",
            neutral_color="#6aa36f",
            key="audio_recorder"
        )
        
        if audio_bytes:
            process_audio(audio_bytes)
        else:
            st.info("👆 Click to start recording (allow mic access).")
    else:
        # Fallback to file uploader
        st.write("**Upload Audio File**")
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Upload a pre-recorded audio file (WAV recommended for best results)"
        )
        
        if uploaded_audio:
            audio_bytes = uploaded_audio.read()
            process_audio(audio_bytes)
        else:
            st.info("👆 Upload an audio file to get started.")

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
        else:
            st.warning("⚠️ Please enter some text first.")

# Troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### Common Issues:
    - **No audio recording?** Try uploading a file instead, or allow mic in browser (HTTPS required).
    - **Transcription fails?** Speak louder, reduce noise, try shorter clips. Use WAV format for uploads.
    - **API key errors?** Ensure `OPENROUTER_API_KEY` is set in .env (local) or in environment variables (deployment).
    - **Module not found?** Check that all packages in requirements.txt are installed.
    - **Deploy fails?** Verify requirements.txt is in root directory. Check deployment logs.
    - **Still stuck?** Share full error logs.
    
    ### Requirements Check:
    """)
    
    # Show which packages are available
    packages_status = {
        "streamlit": True,
        "gtts": True,
        "speech_recognition": True,
        "requests": True,
        "python-dotenv": True,
        "audio-recorder-streamlit": AUDIO_RECORDER_AVAILABLE
    }
    
    for package, status in packages_status.items():
        status_icon = "✅" if status else "❌"
        st.text(f"{status_icon} {package}")

st.markdown("---")
st.caption("💡 Tip: Works best with clear audio and minimal background noise!")
