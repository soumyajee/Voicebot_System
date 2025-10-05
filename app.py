import streamlit as st
import requests
from gtts import gTTS
import os
from dotenv import load_dotenv
import wave
import assemblyai as aai
from time import sleep, time as get_time
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
from scipy.io import wavfile
from noisereduce import reduce_noise

# ------------------------------
# Load API keys
# ------------------------------
# Ensure you have a .env file with OPENROUTER_API_KEY and ASSEMBLYAI_API_KEY
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Set AssemblyAI API key
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY
else:
    aai.settings.api_key = None

# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Local directory for saving audio files
AUDIO_DIR = "./audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

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
    1. **Allow Microphone Access**: When prompted by your browser, click 'Allow' to grant mic access.
    2. **Check Browser Settings**: Go to `chrome://settings/content/microphone` (Chrome) or equivalent to ensure your mic is enabled and selected.
    3. **Use Headphones** if noisy to avoid echo.
    4. Recording starts with the 'Start Recording' button.
    5. Speak clearly 6-12 inches from mic for 5-30s.
    6. **HTTPS Required**: Use HTTPS (locally, use `ngrok http 8501` for secure testing).
    """)

# ------------------------------
# Session State Initialization
# ------------------------------
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
if 'stream_start_time' not in st.session_state:
    st.session_state.stream_start_time = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None  # Stores the latest transcription
if 'warning' not in st.session_state:
    st.session_state.warning = None

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
        temp_file = os.path.join(AUDIO_DIR, f"tts_{int(get_time())}.mp3")
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

def save_as_wav(audio_bytes, sample_rate=16000, channels=1, sample_width=2):
    """Saves raw audio bytes as a WAV file."""
    try:
        filename = f"recording_{int(get_time())}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        frame_size = channels * sample_width
        total_frames = len(audio_bytes) // frame_size
        valid_bytes = audio_bytes[:total_frames * frame_size]
        st.write(f"Debug: Original bytes length: {len(audio_bytes)}, Adjusted to: {len(valid_bytes)} frames")

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(valid_bytes)
        st.write(f"Debug: Saved WAV file at {file_path}, size: {os.path.getsize(file_path)} bytes")
        return file_path
    except Exception as e:
        st.error(f"❌ Error saving WAV file: {e}")
        return None

def check_audio_quality(audio_bytes):
    """Performs a basic check for volume (RMS)."""
    temp_path = save_as_wav(audio_bytes)
    if not temp_path:
        return False
    sample_rate, data = wavfile.read(temp_path)
    os.unlink(temp_path)
    # Convert to 16-bit for RMS calculation if needed, or handle based on dtype
    data = data.astype(np.float64) 
    rms = np.sqrt(np.mean(data**2)) if data.size > 0 else 0
    st.write(f"Debug: Audio RMS (volume): {rms}")
    if rms < 500:  # Threshold can be tuned
        st.warning("⚠️ Audio too quiet—please speak louder and retry.")
        return False
    return True

def transcribe_audio(audio_bytes):
    """Handles audio processing, noise reduction, and AssemblyAI transcription."""
    if not audio_bytes or len(audio_bytes) == 0:
        st.error("❌ No audio data provided. Recording may have failed.")
        return None
    if not aai.settings.api_key:
        st.error("❌ AssemblyAI API key not found. Add ASSEMBLYAI_API_KEY to .env file.")
        return None
        
    file_path = None
    clean_file_path = None
    
    try:
        # 1. Save and validate raw audio
        file_path = save_as_wav(audio_bytes)
        if not file_path:
            st.error("❌ Failed to save audio file for transcription.")
            return None

        file_size = os.path.getsize(file_path)
        st.write(f"Debug: WAV file size: {file_size} bytes")
        
        with open(file_path, "rb") as f:
            st.download_button("Download WAV for Debug", f, file_name=os.path.basename(file_path), mime="audio/wav")

        if file_size < 1000:
            st.error("❌ Audio file too small. Record for at least 5 seconds.")
            return None

        if not check_audio_quality(audio_bytes):
            return None

        # 2. Apply noise reduction
        sample_rate, data = wavfile.read(file_path)
        reduced_noise = reduce_noise(y=data, sr=sample_rate)
        clean_file_path = file_path.replace(".wav", "_clean.wav")
        wavfile.write(clean_file_path, sample_rate, reduced_noise.astype(np.int16))

        # 3. Configure and transcribe
        config = aai.TranscriptionConfig(boost_low_volume=True)
        transcriber = aai.Transcriber()
        
        retries = 3
        for attempt in range(retries):
            try:
                st.write(f"Debug: Attempting transcription (Attempt {attempt + 1}/{retries})...")
                transcript = transcriber.transcribe(clean_file_path, config=config)
                st.write(f"Debug: Transcript status: {transcript.status}")

                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"Transcription error: {transcript.error}")

                text = transcript.text.strip() if hasattr(transcript, 'text') and transcript.text else None
                st.write(f"Debug: Transcript text: '{text}'")

                if text:
                    return text
                else:
                    st.warning("⚠️ Empty transcription. Check mic and recording conditions.")
                    return None
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Transcription HTTP Error: {e}")
                if attempt < retries - 1:
                    sleep(2)
                else:
                    raise
            except Exception as e:
                st.error(f"❌ Transcription Error: {e}")
                if attempt < retries - 1:
                    sleep(2)
                else:
                    raise

    except Exception as e:
        st.error(f"❌ AssemblyAI Transcription Error: {e}")
        return None
    finally:
        # Clean up files
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        if clean_file_path and os.path.exists(clean_file_path):
            os.unlink(clean_file_path)

# Custom Audio Processor for streamlit-webrtc
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []
        self.transcriber = None
        self.last_transcript = ""
        self.rms_history = []

    def recv(self, frame: av.AudioFrame):
        if frame and st.session_state.recording_active:
            # Convert frame to raw bytes (16-bit PCM)
            audio_data = frame.to_ndarray().tobytes()
            self.audio_frames.append(audio_data)

            # Real-time RMS check for volume feedback
            samples = np.frombuffer(audio_data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples**2)) if samples.size > 0 else 0
            self.rms_history.append(rms)
            if len(self.rms_history) > 10:  # Smooth over last 10 frames
                avg_rms = np.mean(self.rms_history[-10:])
                if avg_rms < 500:
                    st.session_state.warning = "⚠️ Audio too quiet—please speak louder."

            # Initialize transcriber if not started
            if not self.transcriber:
                config = aai.RealtimeTranscriberConfig(sample_rate=16000, word_boost=["important", "key"])
                self.transcriber = aai.RealtimeTranscriber(config=config)
                self.transcriber.connect()
                st.session_state.transcription = ""

            # Send audio chunk to AssemblyAI
            if self.transcriber.is_connected():
                self.transcriber.send_audio(audio_data)

            # Update transcription live
            for transcript in self.transcriber.streaming_transcript():
                if transcript and transcript.text:
                    st.session_state.transcription = transcript.text
                    self.last_transcript = transcript.text
        return frame

    def reset(self):
        self.audio_frames = []
        self.rms_history = []
        if self.transcriber and self.transcriber.is_connected():
            self.transcriber.disconnect()
        self.transcriber = None
        st.session_state.transcription = ""
        st.session_state.warning = None

# ------------------------------
# Main Interface Tabs
# ------------------------------
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

# ------------------------------
# Voice Input Tab
# ----------------------
with tab1:
    st.subheader("Browser-Based Voice Recording")
    
    # Initialize WebRTC streamer
    ctx = None
    try:
        ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}],
                "sdpSemantics": "unified-plan"
            }),
            audio_processor_factory=lambda: AudioProcessor(),
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
        )
        if ctx and st.session_state.stream_start_time is None:
            st.session_state.stream_start_time = get_time()
    except Exception as e:
        st.error(f"❌ WebRTC initialization failed: {e}")
        st.write("### Troubleshooting:")
        st.write("- Ensure `streamlit-webrtc` is updated (`pip install streamlit-webrtc --upgrade`).")
        st.write("- Verify microphone permissions are granted.")

    # Check stream state with timeout for permission prompt
    if ctx:
        current_time = get_time()
        if not ctx.state.playing:
            if st.session_state.stream_start_time and (current_time - st.session_state.stream_start_time > 10):
                st.warning("⚠️ Waiting for microphone access. Permission prompt not detected or denied.")
            else:
                st.warning("⚠️ Waiting for microphone access. Please grant permission when prompted by your browser.")
        else:
            st.write("🎙️ Microphone is active. Start recording when ready.")
            st.session_state.stream_start_time = None
            
    # Control recording state with safeguards
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Start Recording", use_container_width=True, disabled=st.session_state.recording_active or not (ctx and ctx.state.playing)):
            st.session_state.recording_active = True
            st.session_state.audio_frames = []
            if ctx and hasattr(ctx, 'audio_processor') and ctx.audio_processor:
                ctx.audio_processor.reset()
            st.session_state.warning = None
            st.rerun()
            
    with col2:
        if st.button("Stop Recording", use_container_width=True, disabled=not st.session_state.recording_active):
            st.session_state.recording_active = False
            if ctx and hasattr(ctx, 'audio_processor') and ctx.audio_processor:
                ctx.audio_processor.reset()
                st.session_state.audio_bytes = b''.join(ctx.audio_processor.audio_frames)
                st.session_state.transcription = ctx.audio_processor.last_transcript
                if st.session_state.audio_bytes:
                    st.success("✅ Recording captured!")
                    st.audio(st.session_state.audio_bytes, format="audio/wav")
                else:
                    st.error("❌ No audio captured. Ensure mic is working.")
            st.rerun()

    with col3:
        if st.button("Clear Recording", use_container_width=True, disabled=st.session_state.recording_active):
            st.session_state.audio_frames = []
            st.session_state.audio_bytes = None
            st.session_state.transcription = None
            st.session_state.warning = None
            if ctx and hasattr(ctx, 'audio_processor') and ctx.audio_processor:
                ctx.audio_processor.reset()
            st.rerun()

    # Live Transcription Display
    st.markdown("---")
    if st.session_state.warning:
        st.warning(st.session_state.warning)
    if st.session_state.transcription:
        st.markdown("### 🎤 Live Transcription:")
        st.info(st.session_state.transcription)
        st.caption("This updates in real-time while recording.")

    # Main Processing Button (for AI response after stopping)
    if st.session_state.transcription and st.button("Process Transcription", type="primary", use_container_width=True, disabled=st.session_state.recording_active):
        with st.spinner("🤔 Getting AI response..."):
            answer = get_response(st.session_state.transcription)
        
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
        else:
            st.error("❌ Failed to get AI response.")
        st.rerun()

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
    ### If recording doesn't work or transcribes incorrectly:
    - **Browser Permissions**: Grant mic access when prompted (check `chrome://settings/content/microphone`).
    - **HTTPS Required**: Use HTTPS (Render enforces this; locally use `ngrok http 8501`).
    - **No Audio Detected**: Speak clearly for 5-10s; test mic in another app.
    - **AssemblyAI Transcription Issues**:
        - Error 422: Download WAV and verify it plays. Ensure 16kHz mono PCM format.
        - Empty transcription: Check for no speech, low volume, wrong format, short duration, or noise.
        - Ensure `ASSEMBLYAI_API_KEY` is valid and not rate-limited.
    - **WebRTC Issues**:
        - Error 'module object is not callable': Update `streamlit-webrtc` (`pip install streamlit-webrtc --upgrade`).
        - Verify compatible browser (e.g., Chrome) and HTTPS.
        - Check console (F12) for errors.
    - **Component Issues**: Update all dependencies to latest versions.
    - **Local Storage**: Ensure `./audio_files/` is writable.
    - **Render Deployment**: Check logs for WebRTC or API errors.
    """)

st.markdown("---")
st.caption("Powered by AssemblyAI & OpenRouter GPT")
