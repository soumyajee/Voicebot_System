import streamlit as st
import requests
from gtts import gTTS
import tempfile
import os
import speech_recognition as sr
from audiorecorder import audiorecorder  # ✅ Corrected import for streamlit-audiorecorder
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
st.write("Record from your browser's microphone! 🎧")

# Microphone setup guide
with st.expander("📱 Microphone Setup (Important!)"):
    st.markdown("""
    ### Before Recording:
    1. **Allow** microphone access when prompted by your browser.
    2. Use Chrome, Firefox, or Edge for best compatibility.
    3. Speak clearly into your device's mic (built-in or external).
    4. If issues: Check browser permissions (site settings > Microphone > Allow).
    """)

# Initialize session state
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

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

# Function to transcribe audio from bytes
def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language="en-US")
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio. Please speak clearly and reduce noise.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

# Main interface tabs
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Live Recording from Browser Microphone")
    
    # Record audio using streamlit-audiorecorder
    audio_bytes = audiorecorder(
        text="Click to record",
        recording_color="#e8b923",
        neutral_color="#6aa36f",
        key="audio_recorder"
    )
    
    if audio_bytes:
        st.success("✅ Recording complete!")
        st.audio(audio_bytes, format="audio/wav")
        
        # Transcribe and get response
        with st.spinner("🔄 Transcribing your speech..."):
            user_input = transcribe_audio(audio_bytes)
            if user_input:
                st.markdown("### 🎤 You said:")
                st.info(f"**\"{user_input}\"**")
                
                with st.spinner("🤔 Getting response..."):
                    answer = get_response(user_input)
                
                if answer:
                    st.markdown("---")
                    st.markdown("### 🤖 Bot Response:")
                    st.write(answer)
                    
                    # Generate audio response
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
    else:
        st.info("👆 Click to start recording (allow mic access).")

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
    ### Common Issues:
    - **No audio recording?** Allow mic in browser (HTTPS required). Test in incognito.
    - **Transcription fails?** Speak louder, reduce noise, try shorter clips. Google STT is rate-limited.
    - **API key errors?** Ensure `OPENROUTER_API_KEY` is set in .env (local) or Render's Environment tab.
    - **Deploy fails?** Verify `requirements.txt` matches below. Check Render logs.
    - **Still stuck?** Share full logs or repo link.
    """)

st.markdown("---")
st.caption("💡 Tip: Audio records client-side. No local mic setup needed!")
