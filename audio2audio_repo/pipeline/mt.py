# pipeline/mt.py
from transformers import pipeline

def load_mt(src_lang, tgt_lang, device=0):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    return pipeline("translation", model=model_name, device=device)

def translate(mt_pipeline, text):
    if not text:
        return ""
    res = mt_pipeline(text)
    if isinstance(res, list) and len(res) > 0:
        return res[0].get("translation_text", "")
    return res.get("translation_text", "")
