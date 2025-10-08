import streamlit as st
import requests
from gtts import gTTS
import os
from dotenv import load_dotenv
import assemblyai as aai
from time import time as get_time
from audio_recorder_streamlit import audio_recorder
import io

# ------------------------------
# Load API keys
# ------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
AUDIO_DIR = "./audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice Bot with Bluetooth Support")
st.write("Connect your Bluetooth headset and start talking! 🎧")

# ------------------------------
# Bluetooth Setup Guide
# ------------------------------
with st.expander("🔵 Bluetooth Audio Setup (Click to Expand)", expanded=False):
    st.markdown("""
    ### 📱 How to Use Bluetooth Devices:
    
    #### **Step 1: Connect Your Bluetooth Device**
    
    **On Windows:**
    1. Open **Settings** → **Bluetooth & devices**
    2. Turn on Bluetooth
    3. Click **Add device** → Select your headset/earbuds
    4. Wait for "Connected" status
    
    **On Mac:**
    1. Click **Apple menu** → **System Settings** → **Bluetooth**
    2. Turn Bluetooth on
    3. Click **Connect** next to your device
    
    **On Mobile (Android/iOS):**
    1. Open **Settings** → **Bluetooth**
    2. Turn on Bluetooth
    3. Tap your device name to connect
    
    #### **Step 2: Set as Default Audio Device**
    
    **On Windows:**
    1. Right-click **speaker icon** in taskbar
    2. Select **Sound settings**
    3. Under **Input**, choose your Bluetooth device
    4. Under **Output**, choose your Bluetooth device
    
    **On Mac:**
    1. Click **Apple menu** → **System Settings** → **Sound**
    2. Select **Input** tab → Choose Bluetooth device
    3. Select **Output** tab → Choose Bluetooth device
    
    **On Browser (Important!):**
    - When the page asks for microphone permission, your browser will use the **system default** audio device
    - Make sure your Bluetooth device is set as default BEFORE opening this page
    
    #### **Step 3: Browser Permissions**
    - Click **Allow** when browser prompts for microphone access
    - Chrome: `chrome://settings/content/microphone`
    - Firefox: `about:preferences#privacy` → Permissions → Microphone
    - Safari: Safari → Settings → Websites → Microphone
    
    #### **💡 Tips for Best Performance:**
    - ✅ Keep Bluetooth device within 30 feet
    - ✅ Ensure device is fully charged
    - ✅ Close other audio apps (Spotify, Zoom, etc.)
    - ✅ Use headsets with built-in mic for clearer audio
    - ✅ Test your setup: Record a short clip first
    
    #### **🔧 Troubleshooting Bluetooth:**
    - **No audio recorded?** Check if Bluetooth is set as default input in system settings
    - **Poor quality?** Try moving closer to your device or reducing interference
    - **Disconnects?** Restart Bluetooth on both devices
    - **Echo/feedback?** Lower output volume or disable mic monitoring
    
    #### **Recommended Bluetooth Devices:**
    - AirPods/AirPods Pro (excellent for iOS/Mac)
    - Sony WH-1000XM series (great noise cancellation)
    - Jabra Elite series (optimized for voice)
    - Any Bluetooth headset with built-in microphone
    """)

# ------------------------------
# Session State Initialization
# ------------------------------
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'bot_response' not in st.session_state:
    st.session_state.bot_response = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

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

def transcribe_audio(audio_bytes):
    """Transcribe audio using AssemblyAI"""
    if not audio_bytes or len(audio_bytes) == 0:
        st.error("❌ No audio data provided.")
        return None
    if not aai.settings.api_key:
        st.error("❌ AssemblyAI API key not found.")
        return None
    
    # Save audio to temporary file
    temp_path = os.path.join(AUDIO_DIR, f"recording_{int(get_time())}.wav")
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size < 1000:
            st.error("❌ Audio too short. Please record for at least 2-3 seconds.")
            return None
        
        # Transcribe with optimized settings for Bluetooth audio
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_code="en",
            punctuate=True,
            format_text=True,
            dual_channel=False,  # Bluetooth is typically mono
            audio_start_from=0,
            audio_end_at=None
        )
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_path, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"❌ Transcription error: {transcript.error}")
            return None
        
        return transcript.text.strip() if transcript.text else None
    except Exception as e:
        st.error(f"❌ Transcription Error: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ------------------------------
# Main Interface Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["🎤 Voice Input", "📝 Text Input", "📜 History"])

# ------------------------------
# Voice Input Tab
# ------------------------------
with tab1:
    st.subheader("Browser-Based Voice Recording")
    
    # Audio device status indicator
    col_status1, col_status2 = st.columns([2, 1])
    with col_status1:
        st.info("🔵 **Bluetooth Ready:** Make sure your device is connected and set as default")
    with col_status2:
        if st.button("🔄 Refresh Page", help="Refresh if you just connected Bluetooth"):
            st.rerun()
    
    st.markdown("---")
    
    # Audio recorder component with custom styling
    st.markdown("""
    <style>
    .audio-recorder {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="🎙️ Click to Record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
        pause_threshold=2.0,  # Auto-stop after 2s of silence
        sample_rate=16000  # Optimal for Bluetooth
    )
    
    # Quick tips
    st.caption("💡 **Tip:** Speak clearly 6-12 inches from your Bluetooth mic. Recording auto-stops after silence.")
    
    if audio_bytes:
        st.success("✅ Audio captured successfully!")
        
        # Audio player
        col_audio, col_info = st.columns([2, 1])
        with col_audio:
            st.audio(audio_bytes, format="audio/wav")
        with col_info:
            st.metric("Audio Size", f"{len(audio_bytes) / 1024:.1f} KB")
        
        # Process button
        if st.button("🎯 Transcribe & Get Response", type="primary", use_container_width=True):
            with st.spinner("🎤 Transcribing your voice..."):
                transcription = transcribe_audio(audio_bytes)
            
            if transcription:
                st.session_state.transcription = transcription
                st.success("✅ Transcription complete!")
                
                # Display transcription
                st.markdown("### 📝 What You Said:")
                st.info(transcription)
                
                # Get AI response
                with st.spinner("🤔 Thinking of a response..."):
                    answer = get_response(transcription)
                
                if answer:
                    st.session_state.bot_response = answer
                    
                    # Save to history
                    st.session_state.conversation_history.append({
                        "user": transcription,
                        "bot": answer,
                        "timestamp": get_time()
                    })
                    
                    st.markdown("### 🤖 Bot Response:")
                    st.write(answer)
                    
                    # Generate TTS
                    with st.spinner("🔊 Generating voice response..."):
                        tts_file = text_to_speech(answer)
                    
                    if tts_file:
                        st.markdown("### 🔊 Listen to Response:")
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.audio(tts_file, format="audio/mp3")
                            st.caption("🔵 Audio will play through your Bluetooth device")
                        with col_b:
                            with open(tts_file, "rb") as f:
                                st.download_button(
                                    label="📥 Download",
                                    data=f,
                                    file_name="response.mp3",
                                    mime="audio/mp3",
                                    use_container_width=True
                                )
                        os.unlink(tts_file)
            else:
                st.error("❌ Failed to transcribe. Check Bluetooth connection and try again.")

# ------------------------------
# Text Input Tab
# ------------------------------
with tab2:
    st.subheader("Text Input (Alternative)")
    user_text = st.text_area("Type your message:", height=150, placeholder="Ask me anything...")
    
    if st.button("Send Text", type="primary"):
        if user_text.strip():
            with st.spinner("🤔 Thinking..."):
                answer = get_response(user_text)
            
            if answer:
                # Save to history
                st.session_state.conversation_history.append({
                    "user": user_text,
                    "bot": answer,
                    "timestamp": get_time()
                })
                
                st.markdown("### 🤖 Bot Response:")
                st.write(answer)
                
                with st.spinner("🔊 Generating voice..."):
                    tts_file = text_to_speech(answer)
                
                if tts_file:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.audio(tts_file, format="audio/mp3")
                        st.caption("🔵 Audio plays through Bluetooth if connected")
                    with col_b:
                        with open(tts_file, "rb") as f:
                            st.download_button(
                                label="📥 Download",
                                data=f,
                                file_name="response.mp3",
                                mime="audio/mp3",
                                use_container_width=True
                            )
                    os.unlink(tts_file)
        else:
            st.warning("Please enter some text first.")

# ------------------------------
# Conversation History Tab
# ------------------------------
with tab3:
    st.subheader("Conversation History")
    
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        st.markdown("---")
        for idx, exchange in enumerate(reversed(st.session_state.conversation_history)):
            with st.container():
                st.markdown(f"**Exchange #{len(st.session_state.conversation_history) - idx}**")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**👤 You:**")
                    st.info(exchange["user"])
                with col2:
                    st.markdown("**🤖 Bot:**")
                    st.success(exchange["bot"])
                st.markdown("---")
    else:
        st.info("No conversations yet. Start recording in the Voice Input tab!")

# ------------------------------
# Bluetooth Testing Section
# ------------------------------
with st.expander("🧪 Test Your Bluetooth Audio"):
    st.markdown("""
    ### Quick Bluetooth Test
    1. Connect your Bluetooth device
    2. Click record below and say "Testing one two three"
    3. Play it back - you should hear it through your Bluetooth device
    """)
    
    test_audio = audio_recorder(
        text="Test Recording",
        recording_color="#f39c12",
        neutral_color="#95a5a6",
        icon_name="flask",
        icon_size="2x",
        key="test_recorder"
    )
    
    if test_audio:
        st.success("✅ Test recording successful!")
        st.audio(test_audio, format="audio/wav")
        st.info("🔵 If you hear this through your Bluetooth device, setup is correct!")

# ------------------------------
# Setup Instructions
# ------------------------------
with st.expander("📦 Installation & Setup"):
    st.code("""
# Install required packages
pip install streamlit
pip install audio-recorder-streamlit
pip install assemblyai
pip install gtts
pip install requests
pip install python-dotenv

# Create .env file with:
OPENROUTER_API_KEY=your_openrouter_key
ASSEMBLYAI_API_KEY=your_assemblyai_key

# Run the app:
streamlit run app.py
    """, language="bash")
    
    st.markdown("""
    ### 🔧 System Requirements:
    - **Python:** 3.8 or higher
    - **Browser:** Chrome, Firefox, Safari, or Edge (latest version)
    - **Bluetooth:** Version 4.0 or higher recommended
    - **Internet:** Required for API calls
    """)

st.markdown("---")
st.caption("🔵 Powered by AssemblyAI & OpenRouter GPT | Bluetooth-Optimized")
