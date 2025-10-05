import streamlit as st
import requests
from gtts import gTTS
import os
from dotenv import load_dotenv
import wave
import assemblyai as aai
from time import sleep
import time
import threading
import io

# Try to import streamlit-webrtc, but provide fallback
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    st.warning("⚠️ streamlit-webrtc not available. Install with: pip install streamlit-webrtc")

# Try to import mic_recorder as fallback
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
    HAS_MIC_RECORDER = False

# Optional: pydub for audio processing (works on Render without PyAudio)
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

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

# Big warning at the top
st.warning("⚠️ **IMPORTANT:** Use the first tab '🎙️ Voice Input (Recommended)' - it's much easier and works immediately!")
st.info("💡 **Avoid the second tab** (WebRTC) unless you know what you're doing - it has complex setup.")

# Microphone setup guide
with st.expander("📱 Microphone Setup (Important!)"):
    st.markdown("""
    ### Before Recording:
    1. **Allow** microphone access in your browser.
    2. **Test** your mic in browser settings.
    3. **Use headphones** if noisy.
    4. Recording is in-browser.
    5. Click 'START' in the WebRTC player, speak for 5-30s, then click 'STOP'.
    6. After stopping, click 'Process Recording' to transcribe.
    """)

# Initialize session state
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
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

def save_as_wav(audio_bytes, sample_rate=48000, channels=1, sample_width=2):
    try:
        if not audio_bytes or len(audio_bytes) < 1000:
            st.error("❌ Audio data too small or empty")
            return None
            
        filename = f"recording_{int(time.time())}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        
        file_size = os.path.getsize(file_path)
        st.write(f"Debug: Saved WAV file at {file_path}, size: {file_size} bytes")
        
        if file_size < 1000:
            st.error("❌ Audio file too small. Record for at least 5 seconds.")
            os.unlink(file_path)
            return None
            
        return file_path
    except Exception as e:
        st.error(f"❌ Error saving WAV file: {e}")
        return None

def transcribe_audio(audio_bytes):
    if not audio_bytes:
        st.error("❌ No audio data provided.")
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
        
        # Offer download for debugging
        with open(file_path, "rb") as f:
            st.download_button(
                "📥 Download WAV for Debug", 
                f, 
                file_name=os.path.basename(file_path), 
                mime="audio/wav",
                key=f"debug_{int(time.time())}"
            )

        # Configure AssemblyAI with better settings
        config = aai.TranscriptionConfig(
            language_code="en",
            speech_model=aai.SpeechModel.best,  # Use best model
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
                
                # Check if transcript has text
                st.write(f"Debug: Transcript status: {transcript.status}")
                st.write(f"Debug: Transcript object: {transcript}")
                st.write(f"Debug: Has text attribute: {hasattr(transcript, 'text')}")
                
                if hasattr(transcript, 'text') and transcript.text:
                    text = transcript.text.strip()
                    st.write(f"Debug: Raw text: '{text}'")
                else:
                    st.error("Debug: Transcript has no text attribute or text is None")
                    text = None
                
                # Filter hallucinations
                if text and text.lower() in ["thank you", "thank you.", "thanks for watching", ""]:
                    st.warning(f"⚠️ Possible hallucination or empty: '{text}'. Try a clearer recording.")
                    os.unlink(file_path)
                    return None
                
                os.unlink(file_path)
                if text:
                    st.success(f"✅ Transcription successful!")
                    st.write(f"Debug: Final transcription: '{text}'")
                    return text
                else:
                    st.warning("⚠️ Empty transcription. Possible issues:")
                    st.markdown("""
                    - **No speech detected** in the audio
                    - **Audio too quiet** - speak louder
                    - **Wrong format** - download the WAV and check if you can hear yourself
                    - **Too short** - record for at least 3-5 seconds
                    - **Background noise** - try in a quieter environment
                    """)
                    return None
                    
            except Exception as e:
                if attempt < retries - 1:
                    st.warning(f"Retrying ({attempt + 1}/{retries})... Error: {str(e)}")
                    sleep(2)
                else:
                    raise e
                    
    except Exception as e:
        st.error(f"❌ AssemblyAI Transcription Error: {e}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        return None

# Custom Audio Processor for streamlit-webrtc
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []
        self.lock = threading.Lock()

    def recv(self, frame: av.AudioFrame):
        # Capture audio frames
        with self.lock:
            sound = frame.to_ndarray()
            self.audio_frames.append(sound.tobytes())
        return frame

    def get_frames(self):
        with self.lock:
            return self.audio_frames.copy()
    
    def reset(self):
        with self.lock:
            self.audio_frames = []
            
    def get_frame_count(self):
        with self.lock:
            return len(self.audio_frames)

# ------------------------------
# Main Interface Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["🎙️ Voice Input (Recommended)", "🎤 Voice Input (WebRTC)", "📝 Text Input"])

# ------------------------------
# Voice Input Tab (Simple Recorder) - NOW FIRST!
# ------------------------------
with tab1:
    st.subheader("Simple Voice Recording ⭐ Recommended")
    st.success("✨ **Easy to use!** Just click 'Start Recording', speak, then 'Stop Recording'. That's it!")
    
    if not HAS_MIC_RECORDER:
        st.error("❌ Mic recorder not available!")
        st.error("**To fix this, run:** `pip install streamlit-mic-recorder`")
        st.code("pip install streamlit-mic-recorder", language="bash")
        st.info("After installing, restart the Streamlit app.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            audio_data = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="🛑 Stop Recording",
                key="recommended_mic_recorder"
            )
        
        with col2:
            if st.button("🔄 Clear Recording", use_container_width=True, key="recommended_clear"):
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
                    st.write(f"Debug: Audio bytes length: {len(audio_bytes)}")
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Auto-process
                    with st.spinner("🔄 Transcribing with AssemblyAI..."):
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
                                                key="recommended_voice_download",
                                                use_container_width=True
                                            )
                                    os.unlink(tts_file)
            except AttributeError:
                st.error("❌ Invalid audio data format. Please try recording again.")

# ------------------------------
# Voice Input Tab (WebRTC) - NOW SECOND
# ------------------------------
with tab2:
    if not HAS_WEBRTC:
        st.error("WebRTC not available. Please use the 'Voice Input (Simple)' tab or install: pip install streamlit-webrtc")
    else:
        st.subheader("Browser-Based Voice Recording (WebRTC)")
        
        st.info("👉 **STEP 1:** Click 'START' in the player below ⬇️ (Grant mic permission if prompted)")
        
        # Initialize WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}
                ]
            }),
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"video": False, "audio": {"sampleRate": 48000}},
            async_processing=True,
        )

        # Debug: Show context state
        st.write(f"Debug: WebRTC context exists: {webrtc_ctx is not None}")
        st.write(f"Debug: Audio processor exists: {webrtc_ctx.audio_processor is not None if webrtc_ctx else False}")
        if webrtc_ctx and webrtc_ctx.audio_processor:
            st.write(f"Debug: Frames captured so far: {webrtc_ctx.audio_processor.get_frame_count()}")

        # Display recording status
        if webrtc_ctx and webrtc_ctx.state.playing:
            st.success("🔴 **STEP 2:** Recording in progress... Speak now! Then click STOP when done.")
        elif webrtc_ctx and webrtc_ctx.audio_processor:
            st.success("✅ **STEP 3:** Recording stopped. Now click '💾 Save & Process Recording' below.")
        else:
            st.warning("⏸️ **Waiting:** Click START in the player above to begin recording.")

        # Control buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Clear Recording", use_container_width=True, key="webrtc_clear"):
                st.session_state.audio_frames = []
                st.session_state.audio_bytes = None
                if webrtc_ctx and webrtc_ctx.audio_processor:
                    webrtc_ctx.audio_processor.reset()
                st.rerun()
        
        with col2:
            if st.button("💾 Save & Process Recording", use_container_width=True, type="primary", key="webrtc_process"):
                if webrtc_ctx is None:
                    st.error("❌ WebRTC context not initialized. Please refresh the page.")
                elif webrtc_ctx.audio_processor is None:
                    st.error("❌ Audio processor not available.")
                    st.warning("🔧 This usually means:")
                    st.markdown("""
                    - You haven't clicked **START** in the WebRTC player yet
                    - The WebRTC connection hasn't been established
                    - Browser didn't grant microphone permissions
                    
                    **Try this:**
                    1. Look for the black WebRTC player box above
                    2. Click the **START** button inside it
                    3. Grant microphone permission when prompted
                    4. Wait 2-3 seconds for connection
                    5. Speak, then click **STOP**
                    6. Then click this button again
                    
                    **Alternative:** Use the "Voice Input (Simple)" tab instead!
                    """)
                else:
                    frames = webrtc_ctx.audio_processor.get_frames()
                    if frames and len(frames) > 0:
                        st.session_state.audio_frames = frames
                        st.session_state.audio_bytes = b''.join(frames)
                        st.success(f"✅ Recording saved! Captured {len(frames)} frames.")
                        st.write(f"Debug: Audio bytes length: {len(st.session_state.audio_bytes)}")
                        
                        # Display audio preview
                        st.markdown("### 🎧 Recorded Audio Preview:")
                        try:
                            st.audio(st.session_state.audio_bytes, format="audio/wav")
                        except:
                            st.warning("⚠️ Audio preview unavailable, but recording was saved.")
                        
                        # Auto-transcribe immediately
                        with st.spinner("🔄 Transcribing with AssemblyAI..."):
                            user_input = transcribe_audio(st.session_state.audio_bytes)
                            
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
                                                    key="webrtc_voice_download",
                                                    use_container_width=True
                                                )
                                        os.unlink(tts_file)
                    else:
                        st.error("❌ No audio frames captured. Ensure you clicked START and spoke into the microphone.")
                        st.info("Steps: 1) Click START, 2) Speak, 3) Click STOP, 4) Click 'Save & Process Recording'")

# ------------------------------
# Voice Input Tab (Simple Recorder)
# ------------------------------
with tab2:
    if not HAS_MIC_RECORDER:
        st.error("Mic recorder not available. Install with: pip install streamlit-mic-recorder")
    else:
        st.subheader("Simple Voice Recording")
        st.info("✨ **Easier alternative!** Just click 'Start Recording', speak, then 'Stop Recording'.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            audio_data = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="🛑 Stop Recording",
                key="simple_mic_recorder"
            )
        
        with col2:
            if st.button("🔄 Clear Recording", use_container_width=True, key="simple_clear"):
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
                    st.write(f"Debug: Audio bytes length: {len(audio_bytes)}")
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Auto-process
                    with st.spinner("🔄 Transcribing with AssemblyAI..."):
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
                                                key="simple_voice_download",
                                                use_container_width=True
                                            )
                                    os.unlink(tts_file)
            except AttributeError:
                st.error("❌ Invalid audio data format. Please try recording again.")

# ------------------------------
# Text Input Tab
# ------------------------------
with tab3:
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
with st.expander("🔧 Troubleshooting & Instructions"):
    st.markdown("""
    ### How to Use (IMPORTANT - Follow in Order):
    1. **Click START** in the WebRTC player (black box with START button)
       - Wait for "Recording in progress..." message
       - Grant microphone permission if prompted
    2. **Speak clearly** for 5-30 seconds
    3. **Click STOP** in the WebRTC player
       - You should see "Frames captured so far: X" increase
    4. **Click '💾 Save & Process Recording'** button
       - This will automatically save, transcribe, and get AI response
       - No need for separate steps!
    
    ### Common Issues:
    
    **"Audio processor not available" Error:**
    - This means you haven't clicked START yet
    - The audio processor is only created after you interact with the WebRTC player
    - **Solution**: Click START, wait 1 second, then proceed
    
    **No Frames Captured:**
    - Check "Debug: Frames captured so far" - it should increase while recording
    - If it stays at 0, microphone isn't working:
      - Check browser permissions (should show 🎤 icon in address bar)
      - Test mic in system settings
      - Try a different browser (Chrome recommended)
      - Ensure HTTPS is enabled (required for mic access)
    
    **Empty or Poor Transcription:**
    - Speak louder and more clearly
    - Record for at least 5-10 seconds
    - Reduce background noise
    - Use headphones to prevent echo
    - Download the WAV file to verify audio quality
    
    **AssemblyAI Issues:**
    - Verify `ASSEMBLYAI_API_KEY` in .env file
    - Check your quota at app.assemblyai.com
    - Wait a few seconds between requests
    
    **WebRTC Not Starting:**
    - Refresh the page
    - Use HTTPS (required for microphone access)
    - Try Chrome or Firefox (Safari may have issues)
    - Check firewall/antivirus settings
    - Clear browser cache
    
    **Technical Details:**
    - Sample Rate: 48kHz (WebRTC default)
    - Format: WAV (mono, 16-bit)
    - Transcription: AssemblyAI API
    - LLM: OpenRouter GPT-4o-mini
    """)

st.markdown("---")
st.caption("Powered by AssemblyAI & OpenRouter GPT | WebRTC Audio Recording")
