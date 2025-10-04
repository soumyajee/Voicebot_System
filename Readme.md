Voice Bot
A Streamlit-based web application that allows users to interact with a chatbot using either text or voice input. The app records audio from any connected microphone (e.g., built-in, USB, or Bluetooth), transcribes it using Google's Speech-to-Text API, queries the OpenRouter API for responses, and generates MP3 audio responses using Google Text-to-Speech (gTTS). The app supports selecting from available audio input devices and provides a simple interface for both text and voice interactions.
Features

Text Input: Type a query and receive a text response with an MP3 audio playback and download option.
Voice Input: Record audio from any connected microphone, transcribe it, and get a text and MP3 response.
Microphone Selection: Choose from a list of available audio input devices.
Dynamic Recording: Select recording durations (3, 5, 7, or 10 seconds).
Error Handling: Robust handling for API, recording, and transcription errors.
User-Friendly Interface: Tabbed layout for text and voice input, with troubleshooting guidance.

Requirements

Python: 3.9 or higher
Operating System: Windows, macOS, or Linux
Microphone: Any connected audio input device (built-in, USB, Bluetooth, etc.)
Internet Connection: Required for OpenRouter API and Google Speech-to-Text API
Dependencies: Listed in requirements.txt (see below)

Installation
1. Clone or Download the Repository
Download the project files to a local directory, e.g., C:\Users\YourUsername\voice_bot\.
2. Set Up a Conda Environment (Recommended)
Create and activate a Conda environment to manage dependencies:
conda create -n voice_bot python=3.9
conda activate voice_bot

3. Install Dependencies
Install the required Python packages using the provided requirements.txt:
pip install -r requirements.txt

Note: On Windows, installing pyaudio may require additional steps:
pip install pipwin
pipwin install pyaudio

On macOS:
brew install portaudio
pip install pyaudio

On Linux:
sudo apt-get install portaudio19-dev
pip install pyaudio

The requirements.txt file contains:
streamlit>=1.38.0
requests>=2.31.0
gtts>=2.5.3
speechrecognition>=3.10.4
pyaudio>=0.2.14

4. Configure Microphone

Connect your microphone (built-in, USB, Bluetooth, etc.).
Ensure it’s recognized in your system settings:
Windows: Settings > System > Sound > Input
macOS: System Preferences > Sound > Input
Linux: Use pavucontrol or system audio settings


Test the microphone to confirm it’s working.

5. Secure API Key
The app uses a hardcoded OpenRouter API key. For production, store the key securely in a .streamlit/secrets.toml file in the project directory:
[general]
OPENROUTER_API_KEY = "your-api-key-here"

Then, update app.py to use:
API_KEY = st.secrets["OPENROUTER_API_KEY"]

Usage

Run the App:
conda activate voice_bot
streamlit run app.py

The app will open in your default browser at http://localhost:8501.

Voice Input:

Go to the Voice Input tab.
Select your microphone from the "Select Microphone" dropdown.
Choose a recording duration (3, 5, 7, or 10 seconds).
Click Start Recording, speak clearly, and wait for the recording to complete.
Click Transcribe & Get Response to transcribe the audio, get a response from OpenRouter, and hear/download the MP3 response.


Text Input:

Go to the Text Input tab.
Type your query in the text area.
Click Send Text to get a text response and MP3 audio playback/download.


Troubleshooting:

If no microphones are listed, ensure your device is connected and recognized in system settings.
If recording fails, check microphone permissions and pyaudio installation.
If transcription fails, speak clearly, reduce background noise, or adjust the recording duration.



Troubleshooting
No Microphones Detected

Ensure your microphone is connected and recognized in system settings.
Run the following to list available devices:import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    device = p.get_device_info_by_index(i)
    if device['maxInputChannels'] > 0:
        print(f"Device {i}: {device['name']}")
p.terminate()


Try a different microphone or USB port.

Recording Errors

Verify pyaudio is installed correctly (pip show pyaudio).
Check system microphone permissions (Windows: Settings > Privacy > Microphone).
Ensure the selected microphone is functional.

Transcription Errors

If you see "Could not understand the audio":
Speak clearly and close to the microphone.
Reduce background noise.
Increase recording duration.
Test the recorded WAV file:import speech_recognition as sr
recognizer = sr.Recognizer()
with sr.AudioFile("path_to_wav_file.wav") as source:
    audio = recognizer.record(source)
    print(recognizer.recognize_google(audio, language="en-US"))





API or TTS Errors

Ensure a stable internet connection for OpenRouter and Google APIs.
Verify the OpenRouter API key is valid.

Security Notes

The hardcoded API key in app.py is insecure for production. Use Streamlit secrets (see Installation step 5).
Ensure microphone access is granted only to trusted applications.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.