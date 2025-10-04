import streamlit as st
import streamlit.components.v1 as components
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
st.write("Record audio directly from your browser!")

# Test microphone access first
st.info("⚠️ **Important:** This app requires HTTPS and microphone permissions. Make sure you're accessing via https:// URL (Streamlit Cloud/Render provide this automatically).")

with st.expander("📱 How to Use"):
    st.markdown("""
    1. Click **Start Recording**
    2. **ALLOW** microphone access in the browser popup
    3. Speak clearly
    4. Click **Stop Recording**
    5. Wait for transcription
    """)

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'last_timestamp' not in st.session_state:
    st.session_state.last_timestamp = 0

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

def transcribe_audio_openrouter(audio_base64):
    if not API_KEY:
        st.error("API key not found.")
        return None
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, 'rb') as audio_file:
            files = {'file': ('audio.webm', audio_file, 'audio/webm')}
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
    st.subheader("Browser-Based Audio Recording")
    
    # Enhanced audio recorder with better debugging
    audio_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
    </head>
    <body>
        <div class="recorder-container">
            <button id="recordButton" class="btn record-btn">🎙️ Start Recording</button>
            <button id="stopButton" class="btn stop-btn" disabled>⏹️ Stop Recording</button>
            <div id="status">Ready to record</div>
            <div id="debug" style="margin-top:10px;font-size:12px;color:#ccc;"></div>
        </div>

        <style>
            body {
                margin: 0;
                padding: 0;
            }
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
            .record-btn {
                background: #ff4444;
                color: white;
            }
            .record-btn:hover:not(:disabled) {
                background: #ff6666;
            }
            .stop-btn {
                background: #4CAF50;
                color: white;
            }
            .stop-btn:hover:not(:disabled) {
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

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            
            const recordButton = document.getElementById('recordButton');
            const stopButton = document.getElementById('stopButton');
            const status = document.getElementById('status');
            const debug = document.getElementById('debug');
            
            // Check if getUserMedia is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                status.textContent = '❌ Browser does not support audio recording';
                status.style.color = '#ff4444';
                debug.textContent = 'getUserMedia not available';
            } else {
                debug.textContent = 'getUserMedia available';
            }
            
            recordButton.addEventListener('click', async () => {
                debug.textContent = 'Requesting microphone access...';
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            sampleRate: 44100
                        } 
                    });
                    
                    debug.textContent = 'Microphone access granted';
                    
                    const options = { mimeType: 'audio/webm' };
                    mediaRecorder = new MediaRecorder(stream, options);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                            debug.textContent = 'Recording data: ' + event.data.size + ' bytes';
                        }
                    };
                    
                    mediaRecorder.onstop = async () => {
                        debug.textContent = 'Processing recording...';
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        debug.textContent = 'Blob size: ' + audioBlob.size + ' bytes';
                        
                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = () => {
                            const base64Audio = reader.result.split(',')[1];
                            const data = {
                                audio: base64Audio,
                                timestamp: Date.now(),
                                size: audioBlob.size
                            };
                            window.parent.postMessage({
                                type: 'streamlit:setComponentValue',
                                value: JSON.stringify(data)
                            }, '*');
                            debug.textContent = 'Sent to Streamlit: ' + audioBlob.size + ' bytes';
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
                    debug.textContent = 'Error: ' + err.message;
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
    </body>
    </html>
    """
    
    audio_response = components.html(audio_html, height=250)
    
    if audio_response:
        try:
            import json
            data = json.loads(audio_response)
            audio_base64 = data.get('audio')
            timestamp = data.get('timestamp', 0)
            
            if audio_base64 and timestamp != st.session_state.last_timestamp:
                st.session_state.last_timestamp = timestamp
                st.session_state.audio_data = audio_base64
                
                audio_bytes = base64.b64decode(audio_base64)
                st.success(f"Recording received: {len(audio_bytes)} bytes")
                st.audio(audio_bytes, format="audio/webm")
                
                with st.spinner("🔄 Transcribing..."):
                    user_input = transcribe_audio_openrouter(audio_base64)
                    
                    if user_input:
                        st.markdown("### 🎤 You said:")
                        st.info(f'"{user_input}"')
                        
                        with st.spinner("🤔 Getting response..."):
                            answer = get_response(user_input)
                        
                        if answer:
                            st.markdown("---")
                            st.markdown("### 🤖 Bot Response:")
                            st.write(answer)
                            
                            with st.spinner("🔊 Generating audio..."):
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
        except Exception as e:
            st.error(f"Error: {e}")

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

with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **Check the debug messages below the recording buttons.**
    
    **If you see "getUserMedia not available":**
    - You MUST use HTTPS (http:// won't work)
    - Streamlit Cloud automatically provides HTTPS
    
    **If you see "Microphone access denied":**
    - Click the lock icon in your browser's address bar
    - Reset permissions for this site
    - Reload the page and allow microphone access
    
    **If nothing happens:**
    - Try Chrome or Firefox (Safari has issues)
    - Check browser console (F12) for errors
    """)

st.markdown("---")
st.caption("Powered by OpenRouter")
