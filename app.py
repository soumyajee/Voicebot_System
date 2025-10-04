import streamlit as st
from audio_recorder_streamlit import audio_recorder
import requests
from gtts import gTTS
import tempfile
import os
import base64
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_AUDIO_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")

def get_response(prompt):
    if not API_KEY:
        st.error("API key not found. Add OPENROUTER_API_KEY to Streamlit secrets.")
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
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def text_to_speech(text):
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def transcribe_audio_openrouter(audio_bytes):
    if not API_KEY:
        st.error("API key not found.")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, 'rb') as audio_file:
            files = {'file': ('audio.wav', audio_file, 'audio/wav')}
            headers = {'Authorization': f'Bearer {API_KEY}'}
            data = {'model': 'openai/whisper-1'}
            
            response = requests.post(
                OPENROUTER_AUDIO_URL,
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
        
        os.unlink(tmp_file_path)
        return result.get('text', '')
        
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Record Your Voice")
    
    # This component handles recording properly
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        
        with st.spinner("🔄 Transcribing your speech..."):
            user_input = transcribe_audio_openrouter(audio_bytes)
            
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
                        audio_file = text_to_speech(answer)
                    
                    if audio_file:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.audio(audio_file, format="audio/mp3")
                        with col_b:
                            with open(audio_file, "rb") as f:
                                st.download_button(
                                    label="📥 Download",
                                    data=f,
                                    file_name="response.mp3",
                                    mime="audio/mp3",
                                    use_container_width=True
                                )
                        os.unlink(audio_file)

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
                    audio_file = text_to_speech(answer)
                
                if audio_file:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.audio(audio_file, format="audio/mp3")
                    with col_b:
                        with open(audio_file, "rb") as f:
                            st.download_button(
                                label="📥 Download",
                                data=f,
                                file_name="response.mp3",
                                mime="audio/mp3",
                                use_container_width=True
                            )
                    os.unlink(audio_file)
        else:
            st.warning("Please enter some text first.")

st.markdown("---")
st.caption("Powered by OpenRouter")
