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

# Try to import streamlit-webrtc, but provide fallback
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    # Streamlit warnings are better than print statements in the app
    # st.warning("⚠️ streamlit-webrtc not available. Install with: pip install streamlit-webrtc")

# Try to import mic_recorder as fallback
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
    HAS_MIC_RECORDER = False

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
    if HAS_MIC_RECORDER or HAS_WEBRTC:
        st.error("AssemblyAI API Key not set. Transcription will fail.")

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
        # Use exponential backoff for requests (good practice)
        for attempt in range(3):
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 429 and attempt < 2:
                sleep(2 ** attempt) # Exponential backoff: 1s, 2s
                continue
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        st.error("❌ OpenRouter API Error: Request failed after multiple retries due to rate limiting.")
        return None
    except requests.RequestException as e:
        st.error(f"❌ OpenRouter API Error: {e}")
        return None

def text_to_speech(text):
    try:
        tts = gTTS(text)
        # Use a unique temporary file path
        temp_file = os.path.join(AUDIO_DIR, f"tts_{os.getpid()}_{int(time.time())}.mp3")
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")
        return None

def save_as_wav(audio_bytes, sample_rate=48000, channels=1, sample_width=2):
    try:
        if not audio_bytes or len(audio_bytes) < 1000:
            st.error("❌ Audio data too small or empty. Please record for at least 3 seconds.")
            return None
            
        filename = f"recording_{os.getpid()}_{int(time.time())}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        
        file_size = os.path.getsize(file_path)
        st.write(f"Debug: Saved WAV file at {file_path}, size: {file_size} bytes")
        
        if file_size < 1000:
            st.error("❌ Audio file too small. Record for at least 3 seconds.")
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
        
    file_path = None
    try:
        file_path = save_as_wav(audio_bytes)
        if not file_path:
            return None

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
                
                # Check if transcript has text
                st.write(f"Debug: Transcript status: {transcript.status}")
                st.write(f"Debug: Transcript object: {transcript}")
                
                if hasattr(transcript, 'text'):
                    text_content = transcript.text.strip()
                    
                    if text_content:
                        text = text_content
                        st.write(f"Debug: Raw text: '{text}'")
                    else:
                        # FIX APPLIED HERE: Explicitly handle the empty string case
                        st.error("Debug: Transcription returned empty text or was just whitespace (likely silence detected).")
                        text = None
                else:
                    st.error("Debug: Transcript object unexpectedly lacks a 'text' attribute.")
                    text = None
                
                # Filter short, common/hallucinated phrases
                if text and len(text.split()) < 3 and text.lower() in ["thank you", "thank you.", "thanks", "i don't know", "yes", "no"]:
                    st.warning(f"⚠️ Possible short input or hallucination: '{text}'. Try a clearer, longer recording.")
                    text = None
                
                # Clean up the audio file
                os.unlink(file_path)
                file_path = None

                if text:
                    st.success(f"✅ Transcription successful!")
                    return text
                else:
                    st.warning("⚠️ Empty transcription. Possible issues:")
                    st.markdown("""
                    - **No speech detected** in the audio
                    - **Audio too quiet** - speak louder
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
        # Ensure cleanup even if transcription fails
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        return None

# Custom Audio Processor for streamlit-webrtc
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []
        self.lock = threading.Lock()

    def recv(self, frame: av.AudioFrame):
        with self.lock:
            # frame.to_ndarray() converts to NumPy array, which is then converted to bytes
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
# 1. Voice Input Tab (Simple Recorder) - RECOMMENDED
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
                            st.markdown("---")
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
# 2. Voice Input Tab (WebRTC)
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
                if webrtc_ctx is None or webrtc_ctx.audio_processor is None:
                    st.error("❌ Audio processor not available. Did you click START?")
                else:
                    frames = webrtc_ctx.audio_processor.get_frames()
                    if frames and len(frames) > 0:
                        st.session_state.audio_frames = frames
                        st.session_state.audio_bytes = b''.join(frames)
                        st.success(f"✅ Recording saved! Captured {len(frames)} frames.")
                        
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
                                st.markdown("---")
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
                        st.error("❌ No audio frames captured. Ensure you clicked START, spoke into the microphone, and then clicked STOP.")

# ------------------------------
# 3. Text Input Tab
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
    ### Why did I get the "Empty Transcription" Error?
    
    The debug line `Debug: Transcript has no text attribute or text is None` was misleading. The underlying problem is that the AssemblyAI service returned the status **completed**, but the resulting text field was an **empty string** (`""`). This means:
    
    * **The AI heard nothing:** The audio file you sent contained only silence or noise below the detection threshold.
    * **The audio was too short:** Very short inputs (1-2 seconds) are often missed.
    * **The audio was too quiet:** Even if you spoke, the microphone signal might have been too weak.
    
    ### How to Fix It:
    
    1.  **Use the Recommended Tab:** Stick to the **"🎙️ Voice Input (Recommended)"** tab for the most reliable recording experience.
    2.  **Speak Clearly and Loudly:** Ensure you speak directly into your microphone.
    3.  **Record Longer:** Always record for at least **3 to 5 seconds** of continuous speech.
    4.  **Check the WAV File:** After recording, click the "📥 Download WAV for Debug" button that appears in the debug output to verify you can actually hear your voice in the saved file. If you can't, your browser/system microphone input is the issue.
    """)

st.markdown("---")
st.caption("Powered by AssemblyAI & OpenRouter GPT | Audio Transcription and LLM Integration")
