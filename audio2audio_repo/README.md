# audio2audio_repo

Simple repository that provides a single-pipeline **Audio → ASR → MT → TTS** translator and a Streamlit demo app.

## Quick start (local)
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run Streamlit:
   ```bash
   streamlit run app_streamlit.py
   ```

## Notes
- The repository uses Hugging Face models (Whisper, Marian/OPUS-MT, and Coqui TTS). For large models, a GPU is recommended.
- Adjust model names inside the pipeline modules as needed.
- This repo is a starter scaffold — for production, separate services, add authentication, rate-limiting, and model caching.

