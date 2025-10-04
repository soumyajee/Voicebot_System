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
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import soundfile as sf

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")

def preprocess_audio_manual(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        audio, sr = librosa.load(tmp_file_path, sr=None)
        
        duration = len(audio) / sr
        if duration < 0.5:
            os.unlink(tmp_file_path)
            st.warning("Audio is too short. Please record a longer clip (at least 0.5s).")
            return None, None
        if np.max(np.abs(audio)) < 0.01:
            os.unlink(tmp_file_path)
            st.warning("Audio is too quiet. Please speak louder or check your microphone.")
            return None, None
        
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, audio, sr)
            with open(tmp_out.name, "rb") as f:
                processed_audio = f.read()
        
        os.unlink(tmp_file_path)
        os.unlink(tmp_out.name)
        
        st.write(f"Preprocessed audio: Duration={duration:.2f}s, Sample rate={sr}Hz, Max amplitude={np.max(np.abs(audio)):.2f}")
        
        return processed_audio, sr
    
    except Exception as e:
        st.error(f"Audio preprocessing error: {e}")
        return None, None

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
        processed_audio, sr = preprocess_audio_manual(audio_bytes)
        if processed_audio is None:
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(processed_audio)
            tmp_file_path = tmp_file.name
        
        audio, sr = librosa.load(tmp_file_path, sr=16000)
        os.unlink(tmp_file_path)
        
        chunk_length = sr  # 1 second
        chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
        
        # Define vocabulary with error handling for missing files
        vocabulary_files = {
            "hello": "samples/hello.wav",
            "world": "samples/world.wav",
            "help": "samples/help.wav",
        }
        vocabulary = {}
        for word, file_path in vocabulary_files.items():
            if os.path.exists(file_path):
                try:
                    audio_data, _ = librosa.load(file_path, sr=16000)
                    vocabulary[word] = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
                except Exception as e:
                    st.warning(f"Failed to load {file_path}: {e}")
            else:
                st.warning(f"Vocabulary file {file_path} not found. Skipping word '{word}'.")
        
        if not vocabulary:
            st.error("No valid vocabulary files found. Please add WAV files to the 'samples/' folder.")
            return "No vocabulary available"
        
        transcription = []
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) < sr * 0.3:
                continue
            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            best_word, best_dist = None, float('inf')
            for word, ref_mfcc in vocabulary.items():
                min_len = min(mfccs.shape[1], ref_mfcc.shape[1])
                mfccs_trunc = mfccs[:, :min_len]
                ref_mfcc_trunc = ref_mfcc[:, :min_len]
                dist, _ = fastdtw(mfccs_trunc.T, ref_mfcc_trunc.T, dist=euclidean)
                st.write(f"Chunk {chunk_idx+1}: DTW distance for '{word}': {dist:.2f}")
                if dist < best_dist and dist < 50:  # Tune threshold
                    best_word, best_dist = word, dist
            if best_word:
                transcription.append(best_word)
        
        return " ".join(transcription).strip() if transcription else "Unknown speech"
        
    except Exception as e:
        st.error(f"Heuristic Transcription error: {e}")
        return None

tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Record Your Voice")
    
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')
        
        with open("debug_audio.wav", "wb") as f:
            f.write(audio_bytes)
        st.write("Saved audio for debugging: debug_audio.wav")
        
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
                                    key="voice_download",
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
                                key="text_download",
                                use_container_width=True
                            )
                    os.unlink(audio_file)
        else:
            st.warning("Please enter some text first.")

st.markdown("---")
st.caption("Powered by OpenRouter")
