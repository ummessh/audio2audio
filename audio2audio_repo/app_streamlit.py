# app_streamlit.py
import streamlit as st
import tempfile, os
from pipeline.asr import load_asr, transcribe
from pipeline.mt import load_mt, translate
from pipeline.tts_coqui import load_tts, synthesize_tts

st.set_page_config(page_title="Audio→Audio Translate", layout="wide")
st.title("Audio → Audio Translator (ASR → MT → TTS)")

uploaded = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"], accept_multiple_files=False)
src_lang = st.text_input("Source language code (e.g. en, fr, es)", value="en")
tgt_lang = st.text_input("Target language code (e.g. fr, en, es)", value="fr")
use_voice_clone = st.checkbox("Use voice cloning (provide short reference clip)", value=False)
ref_clip = None
if use_voice_clone:
    ref_clip = st.file_uploader("Upload short reference audio (3-10s) for voice cloning", type=["wav","mp3","m4a"])

if st.button("Run pipeline"):
    if not uploaded:
        st.error("Upload an audio file to proceed.")
    else:
        tmp_dir = tempfile.mkdtemp()
        in_path = os.path.join(tmp_dir, uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.read())

        st.info("Loading ASR model (may take a moment)...")
        asr = load_asr(model_name="openai/whisper-small", device=-1)  # change device as needed (-1 CPU)

        st.info("Transcribing...")
        asr_out = transcribe(asr, in_path)
        transcript = asr_out.get("text") if isinstance(asr_out, dict) else (asr_out[0].get("text") if asr_out else "")
        st.markdown("**Transcription (source)**")
        st.text_area("transcript", value=transcript, height=160)

        st.info("Loading MT model...")
        try:
            mt = load_mt(src_lang, tgt_lang, device=-1)
            st.info("Translating...")
            translated = translate(mt, transcript)
        except Exception as e:
            translated = transcript
            st.warning(f"MT model unavailable for {src_lang}->{tgt_lang}: {e}\nShowing source transcript instead.")

        st.markdown("**Translated text**")
        st.text_area("translated", value=translated, height=160)

        st.info("Loading TTS model...")
        tts = load_tts(model_name="tts_models/en/ljspeech/tacotron2-DDC")

        out_audio = os.path.join(tmp_dir, f"translated_{tgt_lang}.wav")
        st.info("Synthesizing...")
        speaker_path = None
        if use_voice_clone and ref_clip:
            speaker_path = os.path.join(tmp_dir, "ref.wav")
            with open(speaker_path, "wb") as f:
                f.write(ref_clip.read())

        synthesize_tts(tts, translated, out_path=out_audio, speaker_wav=speaker_path if speaker_path else None)

        st.success("Done — download below")
        st.audio(out_audio, format='audio/wav')
        with open(out_audio, "rb") as f:
            st.download_button("Download translated audio", data=f, file_name=f"translated_{tgt_lang}.wav")
