import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os
import speech_recognition as sr
import pyaudio
import wave
from dotenv import load_dotenv  # ✅ for .env support

# -----------------------------
# Load API key from .env file
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")  # ✅ safely loaded from .env

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")
st.write("Record from any connected microphone! 🎧")

# Microphone setup guide
with st.expander("📱 Microphone Setup (Important!)"):
    st.markdown("""
    ### Before Recording:
    1. **Connect** your microphone (built-in, USB, Bluetooth, etc.) to this device.
    2. **Select** your microphone from the dropdown below.
    3. **Test** your microphone in system settings to ensure it’s working.
    4. **Refresh** this page after connecting a new device.
    """)

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'frames' not in st.session_state:
    st.session_state.frames = []

# Function to get available audio input devices
def get_audio_devices():
    p = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only input devices
                devices.append((device_info['name'], i))
        p.terminate()
        return devices
    except Exception as e:
        st.error(f"❌ Error listing audio devices: {e}")
        return []

# Function to get response from OpenRouter
def get_response(prompt):
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

# Function to record audio
def record_audio(duration=5, device_index=None):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    try:
        p = pyaudio.PyAudio()
        
        # Use selected device or default if None
        if device_index is not None:
            device_info = p.get_device_info_by_index(device_index)
            st.info(f"🎤 Recording from: {device_info['name']}")
        else:
            device_info = p.get_default_input_device_info()
            st.info(f"🎤 Recording from default device: {device_info['name']}")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index
        )
        
        st.session_state.frames = []
        
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            st.session_state.frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save recording
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(st.session_state.frames))
        wf.close()
        
        return temp_file.name
        
    except Exception as e:
        st.error(f"❌ Recording Error: {e}")
        st.write("Please ensure:")
        st.write("- Microphone is connected and selected")
        st.write("- Microphone permissions are granted")
        st.write("- PyAudio is properly installed")
        return None

# Main interface tabs
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Live Recording from Microphone")
    
    # Get available audio devices
    devices = get_audio_devices()
    device_names = [name for name, index in devices] if devices else ["No devices detected"]
    device_indices = [index for name, index in devices] if devices else [None]
    
    # Dropdown for device selection
    selected_device = st.selectbox(
        "Select Microphone",
        device_names,
        index=0,
        help="Choose the microphone to record from. Connect a new device and refresh if it’s not listed."
    )
    
    # Get the device index for the selected device
    selected_device_index = device_indices[device_names.index(selected_device)] if devices else None
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        record_duration = st.selectbox("Recording Duration (seconds)", [3, 5, 7, 10], index=1)
    
    with col2:
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            with st.spinner(f"🔴 Recording for {record_duration} seconds... Speak now!"):
                audio_file = record_audio(duration=record_duration, device_index=selected_device_index)
                if audio_file:
                    st.session_state.audio_file = audio_file
                    st.success("✅ Recording complete!")
                    st.rerun()
    
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.audio_file = None
            st.session_state.frames = []
            st.rerun()
    
    # Process recorded audio
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        st.write("---")
        st.success("✅ Audio recorded successfully!")
        
        # Play recorded audio
        with open(st.session_state.audio_file, "rb") as audio:
            st.audio(audio.read(), format="audio/wav")
        
        # Transcribe button
        if st.button("📝 Transcribe & Get Response", type="primary"):
            with st.spinner("🔄 Transcribing your speech..."):
                try:
                    recognizer = sr.Recognizer()
                    recognizer.energy_threshold = 300
                    recognizer.dynamic_energy_threshold = True
                    
                    with sr.AudioFile(st.session_state.audio_file) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data = recognizer.record(source)
                        
                        user_input = recognizer.recognize_google(audio_data, language="en-US")
                        
                        st.markdown("### 🎤 You said:")
                        st.info(f"**\"{user_input}\"**")
                        
                        # Get bot response
                        with st.spinner("🤔 Getting response..."):
                            answer = get_response(user_input)
                        
                        if answer:
                            st.markdown("---")
                            st.markdown("### 🤖 Bot Response:")
                            st.write(answer)
                            
                            # Generate audio response
                            audio_file = text_to_speech(answer)
                            if audio_file:
                                st.success("🔊 Audio response (will play through your audio device):")
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
                
                except sr.UnknownValueError:
                    st.error("❌ Could not understand the audio. Please:")
                    st.write("- Speak more clearly and slowly")
                    st.write("- Move microphone closer to your mouth")
                    st.write("- Reduce background noise")
                    st.write("- Ensure microphone is properly connected")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

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

# Troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### If recording doesn't work:
    
    **PyAudio Installation:**
    - Windows: `pip install pipwin && pipwin install pyaudio`
    - Mac: `brew install portaudio && pip install pyaudio`
    - Linux: `sudo apt-get install portaudio19-dev && pip install pyaudio`
    
    **Microphone Issues:**
    - Ensure your microphone is connected and selected in the dropdown.
    - Set the microphone as default in System Settings → Sound → Input if needed.
    - Restart the app after connecting a new device.
    - Try a different microphone or USB port if available.
    
    **No audio detected:**
    - Check microphone permissions in your system.
    - Increase recording duration.
    - Speak louder and closer to the mic.
    - Ensure the selected microphone is functioning (test in system settings).
    """)

st.markdown("---")
st.caption("💡 Tip: Select your preferred microphone from the dropdown and test it in system settings!")

