import streamlit as st
import streamlit.components.v1 as components
import requests
from gtts import gTTS
import tempfile
import os
import base64
from dotenv import load_dotenv

# -----------------------------
# Load API key from .env file
# -----------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Add this for Whisper

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="Voice Bot", page_icon="🎙️", layout="wide")

st.title("🎙️ Live Voice Bot")
st.write("Record audio directly from your browser! 🎧")

# Custom audio recorder using HTML5 MediaRecorder API
def audio_recorder_component():
    audio_html = """
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const status = document.getElementById('status');
    
    recordButton.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: base64Audio
                    }, '*');
                };
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            isRecording = true;
            recordButton.disabled = true;
            stopButton.disabled = false;
            status.textContent = '🔴 Recording...';
            status.style.color = '#ff4444';
        } catch (err) {
            status.textContent = '❌ Microphone access denied';
            status.style.color = '#ff4444';
            console.error('Error:', err);
        }
    });
    
    stopButton.addEventListener('click', () => {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            recordButton.disabled = false;
            stopButton.disabled = true;
            status.textContent = '✅ Recording complete!';
            status.style.color = '#00cc00';
        }
    });
    </script>
    
    <style>
        .recorder-container {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        .btn {
            padding: 15px 30px;
            margin: 10px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        #recordButton {
            background: #ff4444;
            color: white;
        }
        #recordButton:hover:not(:disabled) {
            background: #ff6666;
        }
        #stopButton {
            background: #4CAF50;
            color: white;
        }
        #stopButton:hover:not(:disabled) {
            background: #66bb6a;
        }
        #status {
            margin-top: 15px;
            font-size: 18px;
            font-weight: bold;
            color: white;
            min-height: 30px;
        }
    </style>
    
    <div class="recorder-container">
        <button id="recordButton" class="btn">🎙️ Start Recording</button>
        <button id="stopButton" class="btn" disabled>⏹️ Stop Recording</button>
        <div id="status">Ready to record</div>
    </div>
    """
    
    audio_data = components.html(audio_html, height=200)
    return audio_data

# Microphone setup guide
with st.expander("📱 How to Use (Important!)"):
    st.markdown("""
    ### Steps:
    1. Click **"Start Recording"** button below
    2. **Allow** microphone access when browser prompts you
    3. Speak your message clearly
    4. Click **"Stop Recording"** when done
    5. Wait for transcription and response
    
    ### Tips:
    - Use Chrome, Firefox, or Edge (Safari may have issues)
    - Ensure HTTPS connection (required for microphone access)
    - Speak clearly and avoid background noise
    - Keep recordings under 30 seconds for best results
    """)

# Initialize session state
if 'processed_audio' not in st.session_state:
    st.session_state.processed_audio = None

# Function to get response from OpenRouter
def get_response(prompt):
    if not API_KEY:
        st.error("❌ API key not found. Please set OPENROUTER_API_KEY in your environment variables.")
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

# Function to transcribe audio using OpenAI Whisper API
def transcribe_audio_whisper(audio_base64):
    if not OPENAI_API_KEY:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")
        return None
    
    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Call Whisper API
        with open(tmp_file_path, 'rb') as audio_file:
            files = {'file': ('audio.webm', audio_file, 'audio/webm')}
            headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
            data = {'model': 'whisper-1'}
            
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
        
        # Clean up
        os.unlink(tmp_file_path)
        return result['text']
        
    except requests.RequestException as e:
        st.error(f"❌ Whisper API error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

# Main interface tabs
tab1, tab2 = st.tabs(["🎤 Voice Input", "📝 Text Input"])

with tab1:
    st.subheader("Browser-Based Audio Recording")
    
    # Audio recorder component
    audio_data = audio_recorder_component()
    
    # Process audio when received
    if audio_data and audio_data != st.session_state.processed_audio:
        st.session_state.processed_audio = audio_data
        
        # Decode and play audio
        audio_bytes = base64.b64decode(audio_data)
        st.audio(audio_bytes, format="audio/webm")
        
        # Transcribe and get response
        with st.spinner("🔄 Transcribing your speech..."):
            user_input = transcribe_audio_whisper(audio_data)
            
            if user_input:
                st.markdown("### 🎤 You said:")
                st.info(f"**\"{user_input}\"**")
                
                with st.spinner("🤔 Getting AI response..."):
                    answer = get_response(user_input)
                
                if answer:
                    st.markdown("---")
                    st.markdown("### 🤖 Bot Response:")
                    st.write(answer)
                    
                    # Generate audio response
                    with st.spinner("🔊 Generating voice response..."):
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
                
                # Generate audio
                with st.spinner("🔊 Generating voice..."):
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
            st.warning("⚠️ Please enter some text first.")

# Troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### Common Issues:
    
    **Microphone not working?**
    - Make sure you clicked "Allow" when browser asked for mic permission
    - Check browser settings: Site Settings → Microphone → Allow
    - HTTPS is required for microphone access (works on Streamlit Cloud/Render)
    - Try a different browser (Chrome recommended)
    
    **Recording but no transcription?**
    - Speak louder and more clearly
    - Reduce background noise
    - Keep recordings under 30 seconds
    - Try recording again
    
    **API errors?**
    - Ensure `OPENROUTER_API_KEY` is set in environment variables
    - Ensure `OPENAI_API_KEY` is set for Whisper transcription
    - Check your API keys are valid and have credits
    
    **Deployment issues?**
    - This code works on Streamlit Cloud and Render (HTTPS enabled)
    - No PyAudio needed - uses OpenAI Whisper API
    - Uses native browser MediaRecorder API
    """)

st.markdown("---")
st.caption("💡 Powered by HTML5 MediaRecorder + OpenAI Whisper - No PyAudio needed!")
