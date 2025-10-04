def transcribe_audio_openrouter(audio_bytes, model="deepgram/nova-2"):
    if not API_KEY:
        st.error("API key not found.")
        return None
    
    try:
        processed_audio, _ = preprocess_audio_manual(audio_bytes)
        if processed_audio is None:
            return None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(processed_audio)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, 'rb') as audio_file:
            files = {'file': ('audio.wav', audio_file, 'audio/wav')}
            headers = {'Authorization': f'Bearer {API_KEY}'}
            data = {'model': model}
            
            response = requests.post(
                "https://openrouter.ai/api/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
        
        os.unlink(tmp_file_path)
        st.write(f"Transcription response: {result}")
        return result.get('text', '')
        
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None
