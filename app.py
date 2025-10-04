import streamlit as st
from audio_recorder_streamlit import audio_recorder
import requests
from gtts import gTTS
import tempfile
import os
import base64
from dotenv import load_dotenv
import librosa
import numpy as np
import soundfile as sf

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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

def transcribe_audio_heuristic(audio_bytes):
    try:
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Load audio with librosa
        audio, sr = librosa.load(tmp_file_path, sr=16000)  # Resample to 16kHz
        os.unlink(tmp_file_path)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Simple heuristic: Map MFCC patterns to a small vocabulary
        # This is a placeholder; in practice, you'd need a phoneme-to-text mapping
        vocabulary = {
            "hello": np.random.rand(13, 100),  # Dummy MFCC for "hello"
            "world": np.random.rand(13, 100),  # Dummy MFCC for "world"
            # Add more words with precomputed MFCCs for your use case
        }
        
        # Compare MFCCs to vocabulary (simplified distance-based matching)
        transcription = ""
        for word, ref_mfcc in vocabulary.items():
            # Resize MFCCs to match shape for comparison
            min_len = min(mfccs.shape[1], ref_mfcc.shape[1])
            mfccs_trunc = mfccs[:, :min_len]
            ref_mfcc_trunc = ref_mfcc[:, :min_len]
            distance = np.mean((mfccs_trunc - ref_mfcc_trunc) ** 2)
            if distance < 0.1:  # Arbitrary threshold
                transcription += word + " "
        
        if not transcription:
            transcription = "Unknown speech (no matching words found)"
        
        return transcription.strip()
        
    except Exception as e:
        st.error(f"Heuristic Transcription error: {e}")
        return None

tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Record Your Voice")
    
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        
        with st.spinner("🔄 Transcribing your speech..."):
            user_input = transcribe_audio_heuristic(audio_bytes)
            
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
