# pipeline/asr.py
from transformers import pipeline
import torchaudio

def load_asr(model_name="openai/whisper-large-v2", device=0):
    """Load an ASR pipeline. device=-1 for CPU, >=0 for GPU index."""
    return pipeline("automatic-speech-recognition", model=model_name, device=device)

def preprocess_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze().numpy(), target_sr

def transcribe(asr_pipeline, audio_path, **kwargs):
    # Many ASR pipelines accept a path directly (e.g., Whisper)
    out = asr_pipeline(audio_path, **kwargs)
    # Standardize return: either dict or list-of-dicts
    if isinstance(out, list):
        out = out[0]
    return out
