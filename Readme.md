# 🎙️ Universal Voice Bot (Streamlit)

A powerful, real-time voice-to-voice bot application built using Streamlit that works with **ANY audio input device** - built-in microphones, USB mics, wired headsets, or Bluetooth devices. The application transcribes audio using AssemblyAI, sends the text prompt to an OpenRouter large language model (GPT-4o-mini) for a response, and converts that response back into speech using gTTS.

## ✨ Features

- 🔵 **Bluetooth Audio Support**: Seamlessly works with Bluetooth headsets, earbuds, and wireless microphones
- 🎤 **Browser-Based Recording**: Record audio directly using the `audio-recorder-streamlit` component (no WebRTC complexity)
- 📤 **Audio File Upload**: Upload existing audio files (.wav, .mp3, .m4a, .flac, .ogg) for processing
- 💬 **High-Quality Transcription**: Uses AssemblyAI with optimized settings for Bluetooth audio quality
- 🧠 **AI Powered Response**: Integrates with OpenRouter to utilize powerful models like GPT-4o-mini
- 🔊 **Text-to-Speech (TTS)**: Converts AI responses into audible MP3 files using Google Text-to-Speech (gTTS)
- 📜 **Conversation History**: Track all voice and text conversations in one place
- 🔄 **Real-time Feedback**: Shows transcription, AI response, and provides audio playback/download links
- 🧪 **Bluetooth Testing**: Built-in test mode to verify your Bluetooth setup before recording

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| **Frontend/App Framework** | Streamlit |
| **Browser Audio Capture** | audio-recorder-streamlit |
| **Speech-to-Text (STT)** | AssemblyAI |
| **Large Language Model (LLM)** | OpenRouter (GPT-4o-mini) |
| **Text-to-Speech (TTS)** | gTTS |
| **Audio Processing** | scipy, numpy, wave |

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd voice-bot
```

### 2. Install Dependencies
Install all required Python packages using pip:

```bash
pip install streamlit
pip install audio-recorder-streamlit
pip install assemblyai
pip install gtts
pip install requests
pip install python-dotenv
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
This application requires API keys for transcription and the LLM.

Create a file named `.env` in the root directory of your project:

```env
# .env file
OPENROUTER_API_KEY=your_openrouter_api_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

**How to get API keys:**
- **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai) and get your API key
- **AssemblyAI**: Sign up at [assemblyai.com](https://assemblyai.com) and get your API key (free tier available)

### 4. Run the Application
Start the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🔵 Bluetooth Setup Guide

### Step 1: Connect Your Bluetooth Device

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

### Step 2: Set as Default Audio Device

**On Windows:**
1. Right-click **speaker icon** in taskbar
2. Select **Sound settings**
3. Under **Input**, choose your Bluetooth device
4. Under **Output**, choose your Bluetooth device

**On Mac:**
1. Click **Apple menu** → **System Settings** → **Sound**
2. Select **Input** tab → Choose Bluetooth device
3. Select **Output** tab → Choose Bluetooth device

### Step 3: Test Your Setup
1. Open the Voice Bot application
2. Navigate to the **"🧪 Test Your Bluetooth Audio"** section
3. Click record and say "Testing one two three"
4. Play it back - you should hear it through your Bluetooth device
5. If successful, you're ready to use the main Voice Input tab!

## 🌐 Usage Guide

### Voice Input Tab (🎤)

1. **Ensure Bluetooth is Connected**: Check that your device is connected and set as default
2. **Click "🎙️ Click to Record"**: The microphone button will turn red when recording
3. **Speak Clearly**: Position yourself 6-12 inches from your Bluetooth microphone
4. **Recording Auto-Stops**: After 2 seconds of silence, recording automatically stops
5. **Click "🎯 Transcribe & Get Response"**: This will:
   - Transcribe your audio
   - Get AI response
   - Generate voice output (plays through Bluetooth)

### Text Input Tab (📝)

Alternative method if you prefer typing:
1. Type your message in the text area
2. Click "Send Text"
3. Get AI response with voice output

### History Tab (📜)

View all your past conversations (voice and text) in chronological order.

## 📱 Supported Bluetooth Devices

This app works with any Bluetooth audio device, including:
- ✅ AirPods / AirPods Pro / AirPods Max
- ✅ Sony WH-1000XM series
- ✅ Bose QuietComfort series
- ✅ Jabra Elite series
- ✅ Samsung Galaxy Buds
- ✅ Any Bluetooth headset with built-in microphone

## 🔧 Troubleshooting

### Bluetooth Issues

**Problem: No audio recorded**
- ✅ Check if Bluetooth device is connected
- ✅ Verify Bluetooth is set as default input in system settings
- ✅ Refresh the page after connecting Bluetooth
- ✅ Try the test recording feature first

**Problem: Poor audio quality**
- ✅ Move closer to your device (within 30 feet)
- ✅ Reduce interference from other wireless devices
- ✅ Ensure Bluetooth device is fully charged
- ✅ Close other audio applications

**Problem: Echo or feedback**
- ✅ Lower output volume
- ✅ Disable microphone monitoring in headset settings
- ✅ Use headphones instead of speakers

### Transcription Issues

**Problem: Empty transcription**
- ✅ Speak for at least 2-3 seconds
- ✅ Ensure you're speaking clearly and loudly enough
- ✅ Check the audio size indicator (should be > 1KB)
- ✅ Test your microphone in another app first

**Problem: Transcription errors**
- ✅ Speak clearly and avoid background noise
- ✅ Use a quiet environment
- ✅ Check AssemblyAI API key is valid and not rate-limited

### Browser/Permission Issues

**Problem: Microphone permission denied**
- ✅ Chrome: `chrome://settings/content/microphone` → Allow for your site
- ✅ Firefox: `about:preferences#privacy` → Microphone permissions
- ✅ Safari: Safari → Settings → Websites → Microphone
- ✅ Clear browser cache and reload

## 📦 Project Structure

```
voice-bot/
├── app.py                 # Main Streamlit application
├── .env                   # API keys (not committed to git)
├── requirements.txt       # Python dependencies
├── audio_files/          # Temporary audio storage (auto-created)
└── README.md             # This file
```

## 🚀 Deployment

### Streamlit Community Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets (API keys) in the Streamlit dashboard
5. Deploy!

### Other Platforms (Render, Heroku, etc.)
- Ensure you set environment variables for API keys
- The app works on any platform that supports Streamlit
- No special HTTPS setup required (unlike WebRTC solutions)

## 🔐 Security Notes

- ✅ Never commit your `.env` file to version control
- ✅ Add `.env` to your `.gitignore` file
- ✅ Use environment variables for production deployments
- ✅ Regularly rotate your API keys

## 📝 Requirements.txt

```txt
streamlit>=1.28.0
audio-recorder-streamlit>=0.0.8
assemblyai>=0.17.0
gtts>=2.4.0
requests>=2.31.0
python-dotenv>=1.0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [AssemblyAI](https://www.assemblyai.com/) for speech-to-text services
- [OpenRouter](https://openrouter.ai/) for LLM API access
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [audio-recorder-streamlit](https://github.com/stefanrmmr/streamlit_audio_recorder) for the recording component

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the Bluetooth setup guide in the app

---

**Made with ❤️ using Streamlit | 🔵 Bluetooth-Optimized**
