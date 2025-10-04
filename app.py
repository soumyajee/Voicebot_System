import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os
import speech_recognition as sr
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import wave
import numpy as np
import sounddevice as sd  # For local debugging (optional, not needed on Cloud)

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Streamlit page configuration
st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")
st.write("Record voice directly in your browser! 🎧 (Grant mic permission when prompted.)")

# Microphone setup guide
with st.expander("📱 Microphone Setup (Important!)"):
    st.markdown("""
    ### Before Recording:
    1. **Allow** microphone access in your browser (popup will appear on first record).
    2. **Test** your mic in browser settings (e.g., Chrome: chrome://settings/content/microphone).
    3. **Use headphones** to reduce echo or noise.
    4. Recording happens in-browser—no server mic needed.
    5. **Keep recordings short** (5-10s) and speak clearly, close to the mic.
    6. **Download the WAV file** if transcription fails to check audio quality.
    """)

# Initialize session state for audio
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'wav_file_path' not in st.session_state:
    st.session_state.wav_file_path = None

# Function to get response from OpenRouter
def get_response(prompt):
    if not API_KEY:
        st.error("API key not found. Add OPENROUTER_API_KEY to .env file.")
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

# Function to save audio bytes as a valid WAV file
def save_as_wav(audio_bytes, sample_rate=16000, channels=1, sample_width=2):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            with wave.open(tmp_file.name, 'wb') as wf:
                wf.setnchannels(channels)  # Mono
                wf.setsampwidth(sample_width)  # 2 bytes (16-bit)
                wf.setframerate(sample_rate)  # 16kHz
                wf.writeframes(audio_bytes)
            return tmp_file.name
    except Exception as e:
        st.error(f"❌ Error saving WAV file: {e}")
        return None

# Function to analyze audio (for debugging)
def analyze_audio(wav_path):
    try:
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            duration = wf.getnframes() / sample_rate
            audio_data = np.frombuffer(frames, dtype=np.int16)
            max_amplitude = np.max(np.abs(audio_data)) / 32768.0  # Normalize to [-1, 1]
        return {
            "duration_s": duration,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "max_amplitude": max_amplitude
        }
    except Exception as e:
        st.error(f"❌ Error analyzing audio: {e}")
        return None

# Function to transcribe audio bytes
def transcribe_audio(audio_bytes):
    if not audio_bytes:
        return None
    try:
        # Save bytes as a valid WAV file
        tmp_file_path = save_as_wav(audio_bytes)
        if not tmp_file_path:
            return None

        # Debug: Analyze and display audio properties
        audio_info = analyze_audio(tmp_file_path)
        if audio_info:
            st.write(f"Debug: WAV file at {tmp_file_path}, size: {os.path.getsize(tmp_file_path)} bytes")
            st.write(f"Debug: Audio properties: Duration={audio_info['duration_s']:.2f}s, "
                     f"Sample Rate={audio_info['sample_rate_hz']}Hz, "
                     f"Channels={audio_info['channels']}, "
                     f"Sample Width={audio_info['sample_width_bytes']} bytes, "
                     f"Max Amplitude={audio_info['max_amplitude']:.2f}")

        # Provide download link for WAV file
        with open(tmp_file_path, "rb") as f:
            st.download_button(
                label="📥 Download WAV File",
                data=f,
                file_name="recorded_audio.wav",
                mime="audio/wav",
                key="wav_download"
            )

        # Transcribe with Google STT
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 500  # Increased for noisier environments
        recognizer.dynamic_energy_threshold = False  # Disable dynamic adjustment
        recognizer.pause_threshold = 0.8  # Shorter pause to detect speech end

        with sr.AudioFile(tmp_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            user_input = recognizer.recognize_google(audio_data, language="en-US")

        # Cleanup
        os.unlink(tmp_file_path)
        st.session_state.wav_file_path = None
        return user_input
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio. Please try speaking clearly, closer to the mic, or in a quieter environment.")
        os.unlink(tmp_file_path) if os.path.exists(tmp_file_path) else None
        st.session_state.wav_file_path = None
        return None
    except Exception as e:
        st.error(f"❌ Transcription Error: {e}")
        os.unlink(tmp_file_path) if os.path.exists(tmp_file_path) else None
        st.session_state.wav_file_path = None
        return None

# Main interface tabs
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Browser-Based Voice Recording")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Use mic_recorder with supported parameters
        audio_data = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="🛑 Stop Recording",
            key="mic_recorder"
        )
    
    with col2:
        # Clear button
        if st.button("🔄 Clear Recording", use_container_width=True):
            st.session_state.audio_bytes = None
            st.session_state.wav_file_path = None
            st.rerun()
    
    # Process recorded audio
    if audio_data:
        # Extract bytes from dictionary
        try:
            audio_bytes = audio_data.get('bytes', audio_data.get('data', None))
            if not audio_bytes:
                st.error("❌ No audio data found in the recorder output.")
                st.session_state.audio_bytes = None
            else:
                st.session_state.audio_bytes = audio_bytes
                st.success("✅ Recording complete!")
                
                # Debug: Show audio bytes length
                st.write(f"Debug: Audio bytes length: {len(audio_bytes)}")
                
                # Play back the recording
                st.audio(audio_bytes, format="audio/wav")
                
                # Transcribe button
                if st.button("📝 Transcribe & Get Response", type="primary", use_container_width=True):
                    with st.spinner("🔄 Transcribing your speech..."):
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
                                                label="📥 Download Response",
                                                data=f,
                                                file_name="response.mp3",
                                                mime="audio/mp3",
                                                key="voice_download",
                                                use_container_width=True
                                            )
                                    os.unlink(tts_file)
        except AttributeError:
            st.error("❌ Invalid audio data format. Please try recording again.")
            st.session_state.audio_bytes = None

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
                                label="📥 Download Response",
                                data=f,
                                file_name="response.mp3",
                                mime="audio/mp3",
                                key="text_download",
                                use_container_width=True
                            )
                    os.unlink(tts_file)
        else:
            st.warning("Please enter some text first.")

# Troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### If recording doesn't work:
    - **Browser Permissions**: Ensure mic access is granted (check site settings).
    - **HTTPS Required**: Use HTTPS (Streamlit Cloud enforces this).
    - **No Audio Detected**: 
        - Speak clearly and close to the mic.
        - Test mic in another app/site (e.g., voice recorder).
        - Use headphones to reduce noise.
    - **Transcription Fails**: 
        - Keep recordings short (5-10s).
        - Ensure internet for Google STT.
        - Download the WAV file to check if it contains clear speech.
        - Check debug output for audio properties (low amplitude may indicate quiet speech).
    - **Component Issues**: Ensure `streamlit-mic-recorder==0.0.8` is in requirements.txt.
    - **Alternative**: Try `audio-recorder-streamlit` if transcription issues persist.
    - **Streamlit Cloud**: Verify package versions match locally and on Cloud.
    """)

st.markdown("---")
st.caption("Powered by OpenRouter & streamlit-mic-recorder")
