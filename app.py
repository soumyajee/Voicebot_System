import streamlit as st
import requests
from gtts import gTTS
import os
from dotenv import load_dotenv
import wave
import assemblyai as aai
from time import sleep
import time
import io
import threading

# --- Try to Import Necessary Components ---
# Optional: pydub for audio processing (works on Render without PyAudio)
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

# Try to import mic_recorder as main input
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
    HAS_MIC_RECORDER = False
    
# Import for WebRTC is kept commented out for a cleaner, more reliable focus
# try:
#     from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
#     import av
#     HAS_WEBRTC = True
# except ImportError:
#     HAS_WEBRTC = False

# ------------------------------
# Load API keys & Setup
# ------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY
else:
    aai.settings.api_key = None

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

AUDIO_DIR = "./audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Optimized Voice Bot")
st.write("Record voice directly in your browser! Microphone normalization is active. 🎧")

# --- Initial Warnings/Setup Checks ---
if not HAS_MIC_RECORDER:
    st.error("❌ **Dependency Missing:** `streamlit-mic-recorder` is required for voice input.")
    st.code("pip install streamlit-mic-recorder", language="bash")
if not HAS_PYDUB:
    st.warning("⚠️ **Improvement Tip:** `pydub` is recommended for volume normalization to fix empty transcription issues. Install with: `pip install pydub`")
    st.info("Without `pydub`, audio quality will be highly dependent on the user's microphone setup.")

if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

# ------------------------------
# Helper Functions
# ------------------------------

def get_response(prompt):
    if not OPENROUTER_API_KEY:
        st.error("API key not found. Add OPENROUTER_API_KEY to .env.")
        return None
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        st.error(f"❌ OpenRouter API Error: {e}")
        return None

def text_to_speech(text):
    try:
        tts = gTTS(text)
        temp_file = os.path.join(AUDIO_DIR, f"tts_{int(time.time())}.mp3")
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

# --- OPTIMIZED AUDIO SAVE/NORMALIZATION FUNCTION ---
def save_as_wav(audio_bytes, sample_rate=48000, channels=1, sample_width=2):
    """Saves raw audio bytes as WAV, with optional Pydub normalization."""
    if not audio_bytes or len(audio_bytes) < 1000:
        st.error("❌ Audio data too small or empty. Record for at least 2 seconds.")
        return None
        
    filename = f"recording_{int(time.time())}.wav"
    file_path = os.path.join(AUDIO_DIR, filename)
    
    try:
        if HAS_PYDUB:
            # 1. Load raw bytes into pydub AudioSegment
            audio_io = io.BytesIO(audio_bytes)
            # mic_recorder uses a simple 16-bit PCM format
            audio_segment = AudioSegment.from_raw(
                audio_io,
                sample_width=sample_width, 
                frame_rate=sample_rate, 
                channels=channels
            )
            
            # 2. Normalize the volume (key step for transcription reliability)
            st.info("⚙️ Normalizing audio volume for better transcription...")
            normalized_audio = normalize(audio_segment)

            # 3. Export normalized audio to WAV file
            normalized_audio.export(file_path, format="wav")
            
        else:
            # Fallback to direct wave file writing
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

        file_size = os.path.getsize(file_path)
        st.write(f"Debug: Saved WAV file at {file_path}, size: {file_size} bytes")
        
        if file_size < 1000:
            st.error("❌ Audio file too small after processing. Record for at least 5 seconds.")
            os.unlink(file_path)
            return None
            
        return file_path
    except Exception as e:
        st.error(f"❌ Error saving/processing WAV file: {e}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        return None

def transcribe_audio(audio_bytes):
    if not aai.settings.api_key:
        st.error("❌ AssemblyAI API key not found. Add ASSEMBLYAI_API_KEY to .env file.")
        return None
    
    file_path = None
    try:
        file_path = save_as_wav(audio_bytes)
        if not file_path:
            return None

        file_size = os.path.getsize(file_path)
        st.write(f"Debug: WAV file size: {file_size} bytes")
        
        # Configure AssemblyAI with best model for noise tolerance
        config = aai.TranscriptionConfig(
            language_code="en",
            speech_model=aai.SpeechModel.best,  # Best model for noise handling
            punctuate=True,
            format_text=True
        )
        transcriber = aai.Transcriber()
        
        retries = 3
        for attempt in range(retries):
            try:
                st.info(f"🔄 Transcribing with AssemblyAI (attempt {attempt + 1}/{retries})...")
                transcript = transcriber.transcribe(file_path, config=config)
                
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"Transcription error: {transcript.error}")
                
                text = transcript.text.strip() if hasattr(transcript, 'text') and transcript.text else None
                
                # Filter out possible empty/hallucinated text
                if not text or len(text.split()) < 2 or text.lower() in ["thank you", "thanks for watching", ""]:
                    st.warning(f"⚠️ Empty/Short transcription: '{text}'. Try a clearer recording.")
                    if attempt < retries - 1:
                        sleep(2) # Wait before retrying
                        continue # Go to next attempt
                    else:
                        return None
                
                st.success(f"✅ Transcription successful!")
                return text
                
            except Exception as e:
                if attempt < retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{retries})... Error: {str(e)}")
                    sleep(2)
                else:
                    raise e
                    
    except Exception as e:
        st.error(f"❌ AssemblyAI Transcription Error: {e}")
        return None
    finally:
        # Clean up the file
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)

# ------------------------------
# Main Interface Tabs
# ------------------------------
tab1, tab2 = st.tabs(["🎙️ Voice Chat (Recommended)", "📝 Text Chat"])

# ------------------------------
# Voice Input Tab (Simple Recorder) - The Core Feature
# ------------------------------
with tab1:
    st.subheader("🎤 Voice Input Chat")
    st.success("✨ **Easy to use!** Your audio is automatically normalized for better accuracy.")
    
    if not HAS_MIC_RECORDER:
        st.stop()

    col1, col2 = st.columns([1, 1])
    
    with col1:
        audio_data = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="🛑 Stop Recording",
            key="recommended_mic_recorder"
        )
    
    with col2:
        if st.button("🔄 Clear & Reset", use_container_width=True, key="recommended_clear"):
            st.session_state.audio_bytes = None
            st.rerun()
    
    if audio_data and 'bytes' in audio_data:
        audio_bytes = audio_data.get('bytes')
        
        if st.session_state.audio_bytes is None or len(st.session_state.audio_bytes) != len(audio_bytes):
            st.session_state.audio_bytes = audio_bytes
            st.success("✅ Recording complete!")
            st.audio(audio_bytes, format="audio/wav")
            
            # Auto-process
            with st.spinner("🔄 Transcribing and processing audio..."):
                user_input = transcribe_audio(audio_bytes)
                
                if user_input:
                    st.markdown("### 🎤 You Said:")
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
                                        key="recommended_voice_download",
                                        use_container_width=True
                                    )
                            os.unlink(tts_file)
            st.stop() # Stop the rerun cycle after processing

# ------------------------------
# Text Input Tab
# ------------------------------
with tab2:
    st.subheader("📝 Text Input Chat")
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
with st.expander("🔧 Troubleshooting & Audio Quality"):
    st.markdown("""
    ### Why Normalize Audio?
    The primary issue (empty transcription) occurs because the raw audio captured by the browser microphone is often too quiet or contains too much noise/echo for the transcription service to identify clear speech.
    
    By installing **`pydub`** (`pip install pydub`), your app now **automatically normalizes the volume** (`pydub.effects.normalize`) right before sending it to AssemblyAI. This dramatically improves transcription accuracy in noisy or low-volume environments.
    
    ### Troubleshooting Tips:
    * **Always use headphones** 🎧 to prevent speaker echo from being picked up by the microphone.
    * **Install `pydub`** if you haven't yet, as it's the core fix for noisy audio.
    * Ensure your **`ASSEMBLYAI_API_KEY`** and **`OPENROUTER_API_KEY`** are correctly set in your `.env` file and loaded.
    * Record for a minimum of 3-5 seconds.
    """)

st.markdown("---")
st.caption("Optimized with Audio Normalization (via pydub) | Powered by AssemblyAI & OpenRouter")
