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

# Directory for gTTS output (since gTTS requires a file)
AUDIO_DIR = "./audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True) 

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Final Robust Voice Bot")
st.write("Using two-step, decoupled upload and transcription for maximum reliability. 🎧")

# --- Initial Warnings/Setup Checks ---
if not HAS_MIC_RECORDER:
    st.error("❌ **Dependency Missing:** `streamlit-mic-recorder` is required for voice input.")
    st.code("pip install streamlit-mic-recorder", language="bash")
if not HAS_PYDUB:
    st.warning("⚠️ **Critical Library Missing:** `pydub` is required for robust audio processing. Please install it.")
    st.code("pip install pydub", language="bash")
    st.stop() 

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

def process_audio_to_memory(audio_bytes, sample_rate=48000, channels=1, sample_width=2):
    """
    Converts raw audio bytes to a robust, normalized, resampled MP3 stream (in memory).
    Returns an io.BytesIO object containing the MP3 data.
    """
    if not audio_bytes or len(audio_bytes) < 4096:
        st.error("❌ Audio data too small or empty. Record for at least 2 seconds.")
        return None
        
    try:
        # 1. Create a proper WAV file in memory from the raw PCM data
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        wav_buffer.seek(0)
        
        # 2. Use pydub to load, resample, and normalize
        st.info("⚙️ Loading, resampling (48kHz -> 16kHz), and normalizing audio volume (in memory)...")
        audio_segment = AudioSegment.from_file(wav_buffer, format="wav")
        resampled_audio = audio_segment.set_frame_rate(16000)
        normalized_audio = normalize(resampled_audio)

        # 3. Export the processed audio to a new in-memory MP3 buffer
        mp3_buffer = io.BytesIO()
        normalized_audio.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0) # Rewind buffer to start for reading by AssemblyAI

        buffer_size = len(mp3_buffer.getvalue())
        st.write(f"Debug: In-memory MP3 buffer size: {buffer_size} bytes")
        
        if buffer_size < 1024:
            st.error("❌ Processed audio stream still too small.")
            return None
            
        return mp3_buffer
    except Exception as e:
        st.error(f"❌ Critical Error processing audio stream: {e}. Ensure PyDub/FFmpeg are available.")
        return None

def transcribe_audio(audio_bytes):
    """Handles transcription using manual upload via requests, then transcribes the URL."""
    if not aai.settings.api_key:
        st.error("❌ AssemblyAI API key not found.")
        return None
    
    audio_stream = None
    try:
        audio_stream = process_audio_to_memory(audio_bytes)
        if not audio_stream:
            return None

        # --- STEP 1: Manual Upload (Decoupling from SDK) ---
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {"authorization": aai.settings.api_key}
        
        st.info("⬆️ Uploading processed audio stream directly (manual requests method)...")
        upload_response = requests.post(
            upload_url,
            headers=headers,
            # Read the bytes from the stream for the request body
            data=audio_stream.read() 
        )
        upload_response.raise_for_status() # Check for HTTP errors (4xx/5xx)
        uploaded_audio_url = upload_response.json().get('upload_url')
        
        if not uploaded_audio_url:
            raise Exception("Upload succeeded but returned no 'upload_url'.")

        # --- STEP 2: Transcribe the URL ---
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
                st.info(f"🔄 Transcribing URL with AssemblyAI (attempt {attempt + 1}/{retries})...")
                # Pass the temporary URL back to the SDK for transcription
                transcript = transcriber.transcribe(uploaded_audio_url, config=config) 
                
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"Transcription error: {transcript.error}")
                
                text = transcript.text.strip() if transcript.text else None
                
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
                # Retry for transcription submission/processing errors
                if attempt < retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{retries})... Error: {str(e)}")
                    sleep(2)
                else:
                    raise e
                    
    except requests.RequestException as e:
        # Catch specific manual upload request errors
        st.error(f"❌ Critical Upload Failure: Could not reach AssemblyAI upload endpoint. Error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ AssemblyAI Processing Failure: {e}")
        return None
    
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
        
        if st.session_state.audio_bytes is None or len(st.session_state.audio_bytes) != len(audio_bytes):
            st.session_state.audio_bytes = audio_bytes
            st.success("✅ Recording complete! Processing audio...")
            st.audio(audio_bytes, format="audio/wav") 
            
            with st.spinner("🔄 Transcribing and getting AI response..."):
                user_input = transcribe_audio(audio_bytes)
                
                if user_input:
                    st.markdown("### 🎤 You Said:")
                    st.info(f'"{user_input}"')
                    
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
            st.stop() 

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
    ### Solving the Persistent Upload Error
    The latest fix uses a **two-step, decoupled process** to upload and transcribe:
    1.  **Manual Upload:** The processed MP3 audio stream is sent directly to AssemblyAI's `v2/upload` API endpoint using the stable Python `requests` library. This bypasses potential instability in the SDK's internal stream handling.
    2.  **URL Transcription:** The resulting public URL from the upload is then passed to the AssemblyAI SDK for transcription.
    
    This technique is the most robust way to handle file uploads in complex cloud environments like Streamlit Cloud.
    
    ### Installation Check
    You still need **FFmpeg** installed on the host system (required by `pydub`) for audio processing.
    
    Dependencies:
    * `pip install streamlit`
    * `pip install streamlit-mic-recorder`
    * `pip install assemblyai`
    * `pip install python-dotenv`
    * `pip install gTTS`
    * `pip install pydub` **(Crucial)**
    """)

st.markdown("---")
st.caption("Optimized with Decoupled Upload/Transcription Pipeline | Powered by AssemblyAI & OpenRouter")
