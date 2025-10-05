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
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
    HAS_MIC_RECORDER = False

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

st.title("🎙️ Robust Voice Bot")
st.write("Now using MP3 encoding for maximum transcription compatibility. 🎧")

# --- Initial Warnings/Setup Checks ---
if not HAS_MIC_RECORDER:
    st.error("❌ **Dependency Missing:** `streamlit-mic-recorder` is required for voice input.")
    st.code("pip install streamlit-mic-recorder", language="bash")
if not HAS_PYDUB:
    st.warning("⚠️ **Critical Library Missing:** `pydub` is required for robust audio processing. Please install it.")
    st.code("pip install pydub", language="bash")
    st.stop() # Stop execution if critical dependency is missing

if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

# ------------------------------
# Helper Functions
# ------------------------------

def get_response(prompt):
    """Fetches AI response from OpenRouter."""
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
    """Generates an MP3 file from text using gTTS."""
    try:
        tts = gTTS(text)
        temp_file = os.path.join(AUDIO_DIR, f"tts_{int(time.time())}.mp3")
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

# --- CRITICAL FIX: ROBUST MP3 CONVERSION AND NORMALIZATION ---
def save_for_transcription(audio_bytes, sample_rate=48000, channels=1, sample_width=2):
    """
    Converts raw audio bytes to a robust, normalized, resampled MP3 file.
    It ensures a proper WAV header, then uses pydub to resample, normalize, 
    and save the resulting stream as MP3 for maximum transcription compatibility.
    
    Default mic-recorder output: 48000 Hz, 1 channel, 16-bit (sample_width=2).
    """
    if not audio_bytes or len(audio_bytes) < 4096: # Minimum 4KB expected for a short recording
        st.error("❌ Audio data too small or empty. Record for at least 2 seconds.")
        return None
        
    filename = f"recording_{int(time.time())}.mp3"
    file_path = os.path.join(AUDIO_DIR, filename)

    try:
        # 1. Create a proper WAV file in memory (BytesIO) from the raw PCM data
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        
        buffer.seek(0)
        
        # 2. Use pydub to load the complete WAV file from the in-memory buffer
        st.info("⚙️ Loading, resampling (48kHz -> 16kHz), and normalizing audio volume...")
        audio_segment = AudioSegment.from_file(buffer, format="wav")
        
        # 3. Resample to 16kHz (Standard for high-accuracy speech transcription)
        resampled_audio = audio_segment.set_frame_rate(16000)
        
        # 4. Normalize the volume to combat quiet recordings
        normalized_audio = normalize(resampled_audio)

        # 5. Export the processed audio to a physical MP3 file (The new compatibility fix)
        normalized_audio.export(file_path, format="mp3")
        
        file_size = os.path.getsize(file_path)
        st.write(f"Debug: Saved MP3 file size: {file_size} bytes (resampled to 16kHz)")
        
        if file_size < 1024: # MP3 files are much smaller, checking for 1KB minimum
            st.error("❌ Processed audio file still too small. Try recording louder or longer.")
            os.unlink(file_path)
            return None
            
        return file_path
    except Exception as e:
        st.error(f"❌ Critical Error saving/processing MP3 file: {e}. Check if PyDub/FFmpeg dependencies are met.")
        if os.path.exists(file_path):
            os.unlink(file_path)
        return None

def transcribe_audio(audio_bytes):
    """Handles transcription with AssemblyAI, including file processing and cleanup."""
    if not aai.settings.api_key:
        st.error("❌ AssemblyAI API key not found. Add ASSEMBLYAI_API_KEY to .env file.")
        return None
    
    file_path = None
    try:
        file_path = save_for_transcription(audio_bytes)
        if not file_path:
            return None

        # Configure AssemblyAI for best transcription results
        config = aai.TranscriptionConfig(
            language_code="en",
            speech_model=aai.SpeechModel.best,
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
                
                text = transcript.text.strip() if transcript.text else None
                
                # Check for empty or non-speech transcriptions
                if not text or len(text.split()) < 2:
                    st.warning(f"⚠️ Empty/Short transcription: '{text}'. Retrying if possible.")
                    if attempt < retries - 1:
                        sleep(1) 
                        continue 
                    else:
                        st.error("❌ Failed to get meaningful transcription after all attempts.")
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
        st.error(f"❌ AssemblyAI Processing Failure: {e}")
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
# Voice Input Tab 
# ------------------------------
with tab1:
    st.subheader("🎤 Voice Input Chat")
    st.success("✨ **Optimized for noisy environments!** Use this tab for best results.")
    
    if not HAS_MIC_RECORDER or not HAS_PYDUB:
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
        
        # Only process if new audio bytes are received
        if st.session_state.audio_bytes is None or len(st.session_state.audio_bytes) != len(audio_bytes):
            st.session_state.audio_bytes = audio_bytes
            st.success("✅ Recording complete! Processing audio...")
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
    
    user_text = st.text_area(
        "Type your message:", 
        height=150, 
        placeholder="Ask me anything...", 
        key="text_input_area"
    )
    
    if st.button("Send Text", type="primary", key="send_text_button"):
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
    ### MP3 Conversion and Resampling (New)
    The code now performs three crucial steps before transcription:
    1. **Robust WAV Header:** Ensures a valid audio stream is created from the raw mic input.
    2. **16kHz Resampling & Normalization:** The audio is normalized for volume and resampled to the optimal **16kHz**.
    3. **MP3 Export:** The processed audio is saved as an **MP3** file. This highly compatible, compressed format should minimize conflicts with the transcription service's ingestion process, finally eliminating the "Empty Transcription" error.
    
    ### Installation Check
    For this app to run correctly in deployment environments (like Streamlit Cloud), you must ensure `pydub`'s underlying dependency, **FFmpeg**, is installed on the system path.
    
    Dependencies:
    * `pip install streamlit`
    * `pip install streamlit-mic-recorder`
    * `pip install assemblyai`
    * `pip install python-dotenv`
    * `pip install gTTS`
    * `pip install pydub` **(Crucial)**
    """)

st.markdown("---")
st.caption("Optimized with Audio Normalization (via pydub), Robust WAV Header Generation, and 16kHz Resampling | Powered by AssemblyAI & OpenRouter")
