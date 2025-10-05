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

if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
if 'stream_start_time' not in st.session_state:
    st.session_state.stream_start_time = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None

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
    try:
        filename = f"recording_{int(get_time())}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        # Calculate expected frame size in bytes
        frame_size = channels * sample_width
        total_frames = len(audio_bytes) // frame_size
        valid_bytes = audio_bytes[:total_frames * frame_size]  # Truncate to whole frames
        st.write(f"Debug: Original bytes length: {len(audio_bytes)}, Adjusted to: {len(valid_bytes)} frames")

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(valid_bytes)
        st.write(f"Debug: Saved WAV file at {file_path}, size: {os.path.getsize(file_path)} bytes")
        with wave.open(file_path, 'rb') as wf_read:
            st.write(f"Debug: WAV sample rate: {wf_read.getframerate()}, channels: {wf_read.getnchannels()}")
        return file_path
    except Exception as e:
        st.error(f"❌ Error saving WAV file: {e}")
        return None

def check_audio_quality(audio_bytes):
    temp_path = save_as_wav(audio_bytes)
    if not temp_path:
        return False
    sample_rate, data = wavfile.read(temp_path)
    os.unlink(temp_path)
    rms = np.sqrt(np.mean(data**2)) if data.size > 0 else 0
    st.write(f"Debug: Audio RMS (volume): {rms}")
    if rms < 500:
        st.warning("⚠️ Audio too quiet—please speak louder and retry.")
        return False
    return True

def transcribe_audio(audio_bytes):
    if not audio_bytes or len(audio_bytes) == 0:
        st.error("❌ No audio data provided. Recording may have failed.")
        return None
    if not aai.settings.api_key:
        st.error("❌ AssemblyAI API key not found. Add ASSEMBLYAI_API_KEY to .env file.")
        return None
    try:
        file_path = save_as_wav(audio_bytes)
        if not file_path:
            return None

        file_size = os.path.getsize(file_path)
        st.write(f"Debug: WAV file size: {file_size} bytes")
        
        with open(file_path, "rb") as f:
            st.download_button("Download WAV for Debug", f, file_name=os.path.basename(file_path), mime="audio/wav")

        if file_size < 1000:
            st.error("❌ Audio file too small. Record for at least 5 seconds.")
            os.unlink(file_path)
            return None

        if not check_audio_quality(audio_bytes):
            os.unlink(file_path)
            return None

        sample_rate, data = wavfile.read(file_path)
        st.write(f"Debug: Read sample rate: {sample_rate}, data length: {len(data)}")
        reduced_noise = reduce_noise(y=data, sr=sample_rate)
        clean_file_path = file_path.replace(".wav", "_clean.wav")
        wavfile.write(clean_file_path, sample_rate, reduced_noise.astype(np.int16))

        config = aai.TranscriptionConfig(boost_low_volume=True)
        transcriber = aai.Transcriber()
        
        retries = 3
        for attempt in range(retries):
            try:
                transcript = transcriber.transcribe(clean_file_path)
                st.write(f"Debug: Transcript status: {transcript.status}")
                st.write(f"Debug: Transcript object: {transcript}")
                st.write(f"Debug: Has text attribute: {hasattr(transcript, 'text')}")
                text = transcript.text.strip() if hasattr(transcript, 'text') and transcript.text else None
                st.write(f"Debug: Transcript text: '{text}'")
                
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"Transcription error: {transcript.error}")
                
                if text and text.lower() in ["thank you", "thank you.", "thanks for watching"]:
                    st.warning(f"⚠️ Possible hallucination detected: '{text}'. Proceeding anyway.")
                
                os.unlink(file_path)
                os.unlink(clean_file_path)
                if text:
                    st.write(f"Debug: Transcription successful: '{text}'")
                    return text
                else:
                    st.warning("⚠️ Empty transcription. Possible issues:")
                    st.write("- No speech detected in the audio")
                    st.write("- Audio too quiet - speak louder")
                    st.write("- Wrong format - download the WAV and check if you can hear yourself")
                    st.write("- Too short - record for at least 5 seconds")
                    st.write("- Background noise - try in a quieter environment")
                    return None
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 422:
                    st.error(f"❌ Critical Upload Failure: 422 Unprocessable Entity. The audio file may be invalid. Download the WAV to verify.")
                    with open(file_path, "rb") as f:
                        st.download_button("Download WAV for Manual Upload", f, file_name=os.path.basename(file_path), mime="audio/wav")
                else:
                    st.error(f"❌ Transcription Error: {e}")
                if attempt < retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{retries})...")
                    sleep(2)
                else:
                    raise
            except Exception as e:
                st.error(f"❌ Transcription Error: {e}")
                if attempt < retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{retries})...")
                    sleep(2)
                else:
                    raise
    except Exception as e:
        st.error(f"❌ AssemblyAI Transcription Error: {e}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        if os.path.exists(clean_file_path):
            os.unlink(clean_file_path)
        return None

# Custom Audio Processor for streamlit-webrtc
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame):
        if frame and st.session_state.recording_active:
            self.audio_frames.append(frame.to_ndarray().tobytes())
            st.write(f"Debug: Received audio frame, total frames: {len(self.audio_frames)}")
        return frame

    def reset(self):
        self.audio_frames = []

# ------------------------------
# Main Interface Tabs
# ------------------------------
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

# ------------------------------
# Voice Input Tab
# ----------------------
with tab1:
    st.subheader("Browser-Based Voice Recording")
    
    # Initialize WebRTC streamer with SENDRECV mode
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
        st.write("- Use a compatible browser (e.g., Chrome).")
        st.write("- Check for HTTPS (use `ngrok http 8501` locally).")
        st.write("- Review console (F12) for WebRTC errors.")

    # Check stream state with timeout for permission prompt
    if ctx:
        current_time = get_time()
        if not ctx.state.playing:
            if st.session_state.stream_start_time and (current_time - st.session_state.stream_start_time > 10):
                st.warning("⚠️ Waiting for microphone access. Permission prompt not detected or denied.")
                st.write("### Troubleshooting Steps:")
                st.write("- Click the lock icon in the address bar and ensure 'Microphone' is set to 'Allow'.")
                st.write("- Reload the page and grant access again.")
                st.write("- Use a secure connection (HTTPS) or run locally with `ngrok http 8501`.")
                st.write("- Check browser console (F12) for WebRTC errors.")
            else:
                st.warning("⚠️ Waiting for microphone access. Please grant permission when prompted by your browser.")
        else:
            st.write("🎙️ Microphone is active. Start recording when ready.")
            st.session_state.stream_start_time = None  # Reset timer when stream starts

    # Control recording state with safeguards
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Recording", use_container_width=True) and not st.session_state.recording_active:
            st.session_state.recording_active = True
            st.session_state.audio_frames = []
            if ctx and hasattr(ctx, 'audio_processor') and ctx.audio_processor:
                ctx.audio_processor.reset()
            else:
                st.session_state.audio_frames = []  # Fallback reset
            st.rerun()
    with col2:
        if st.button("Stop Recording", use_container_width=True) and st.session_state.recording_active:
            st.session_state.recording_active = False
            if ctx and hasattr(ctx, 'audio_processor') and ctx.audio_processor and ctx.audio_processor.audio_frames:
                st.session_state.audio_frames = ctx.audio_processor.audio_frames
                st.session_state.audio_bytes = b''.join(st.session_state.audio_frames)
                if len(st.session_state.audio_frames) < 10:
                    st.error("❌ Too few audio frames captured. Record for at least 5 seconds.")
                else:
                    st.success("✅ Recording processed!")
                    st.write(f"Debug: Audio bytes length: {len(st.session_state.audio_bytes)}")
                    st.audio(st.session_state.audio_bytes, format="audio/wav")
            else:
                st.error("❌ No audio frames captured. Ensure mic is working and permissions are granted.")
            st.rerun()

    if st.button("🔄 Clear Recording", use_container_width=True):
        st.session_state.audio_frames = []
        st.session_state.audio_bytes = None
        st.session_state.recording_active = False
        st.session_state.transcription = None
        if ctx and hasattr(ctx, 'audio_processor') and ctx.audio_processor:
            ctx.audio_processor.reset()
        else:
            st.session_state.audio_frames = []  # Fallback reset
        st.rerun()

    if st.session_state.audio_bytes and st.button("Process Recording", type="primary", use_container_width=True):
        with st.spinner("🔄 Transcribing with AssemblyAI..."):
            user_input = transcribe_audio(st.session_state.audio_bytes)
            st.session_state.transcription = user_input  # Store transcription
            
            if user_input:
                st.markdown("### 🎤 Transcribed Text:")
                st.info(f'"{user_input}"')  # Prominently display transcription
                st.write("📋 Copy the transcription above for use.")
                
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
            else:
                st.session_state.transcription = None
                st.error("❌ Transcription failed. Check debug logs and troubleshooting section.")

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
        - Error "not a whole number of frames": Record longer or adjust mic settings.
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
