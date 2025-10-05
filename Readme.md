🎙️ Live Voice Bot (Streamlit)

This is a powerful, real-time voice-to-voice bot application built using Streamlit that allows users to record their voice directly from the browser (via streamlit-webrtc) or upload an audio file. The application then transcribes the audio using AssemblyAI, sends the text prompt to an OpenRouter large language model (GPT-4o-mini) for a response, and converts that response back into speech using gTTS.
✨ Features

    🎤 Live Browser Recording: Record audio directly using the streamlit-webrtc component.

    📤 Audio File Upload: Upload existing .wav or .mp3 files for processing.

    💬 High-Quality Transcription: Uses AssemblyAI for accurate Speech-to-Text conversion, including noise reduction for cleaner results.

    🧠 AI Powered Response: Integrates with OpenRouter to utilize powerful models like GPT-4o-mini for generating intelligent text responses.

    🔊 Text-to-Speech (TTS): Converts the AI's text response into an audible MP3 file using Google Text-to-Speech (gTTS).

    🔄 Real-time Feedback: Shows transcription, AI response, and provides audio playback/download links.

🛠️ Technologies Used

    Frontend/App Framework: Streamlit

    Live Audio Capture: streamlit-webrtc

    Speech-to-Text (STT): AssemblyAI

    Large Language Model (LLM): OpenRouter (using openai/gpt-4o-mini)

    Text-to-Speech (TTS): gTTS

    Audio Processing: scipy.io.wavfile, numpy, noisereduce, wave, av

⚙️ Setup and Installation
1. Clone the Repository (or save the file)

git clone <your-repo-link>
cd <your-repo-name>

(If you are running this locally, save the provided Python code as app.py)
2. Install Dependencies

You can install all required Python packages using pip:

pip install streamlit requests gtts python-dotenv assemblyai numpy scipy noisereduce streamlit-webrtc av

3. Set Up API Keys

This application requires API keys for transcription and the LLM.

    Create a file named .env in the root directory of your project.

    Add your API keys to the file:

    # .env file
    OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY_HERE"
    ASSEMBLYAI_API_KEY="YOUR_ASSEMBLYAI_API_KEY_HERE"

4. Run the Application

Start the Streamlit application from your terminal:

streamlit run app.py

🌐 Usage Notes (Crucial for Live Recording)
⚠️ HTTPS Requirement

The streamlit-webrtc component, which handles microphone access, requires a secure connection (HTTPS).

    Deployment: If you deploy to platforms like Streamlit Community Cloud or Render, HTTPS is handled automatically.

    Local Testing: If testing locally, you must use a tool like ngrok to expose your local Streamlit port (default 8501) over HTTPS:

    ngrok http 8501

    Then, access the app using the HTTPS URL provided by ngrok.

🎤 Microphone Permissions

    Grant Access: You must click "Allow" when your browser prompts you for microphone permission.

    Troubleshooting: If the stream does not start, check the lock icon in your browser's address bar to ensure microphone access is explicitly set to "Allow" for the page's domain (or ngrok URL).

Recording Steps

    Navigate to the "🎤 Live Recording" tab.

    Wait for the status to change to "🎙️ Microphone is active."

    Click "Start Recording" and speak clearly for at least 5 seconds.

    Click "Stop Recording".

    Click "Transcribe Recording" to trigger the full workflow (STT, LLM, TTS).
