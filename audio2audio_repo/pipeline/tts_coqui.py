# pipeline/tts_coqui.py
# Uses Coqui TTS API. Install via `pip install TTS` (version may vary).
from TTS.api import TTS
import os

def load_tts(model_name="tts_models/en/ljspeech/tacotron2-DDC"):
    """Load a TTS model. Replace with an XTTS/Coqui voice-cloning model for advanced use."""
    tts = TTS(model_name)
    return tts

def synthesize_tts(tts, text, out_path="out.wav", speaker_wav=None):
    if speaker_wav:
        # Some XTTS models accept speaker_wav for cloning; may vary by model.
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=out_path)
    else:
        tts.tts_to_file(text=text, file_path=out_path)
    return out_path
