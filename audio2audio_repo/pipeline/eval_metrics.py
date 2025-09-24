# pipeline/eval_metrics.py
import evaluate

wer_metric = evaluate.load("wer")
sacrebleu = evaluate.load("sacrebleu")
# COMET model may require additional setup and credentials; optional
# comet = evaluate.load("comet")

def compute_wer(refs, hyps):
    return wer_metric.compute(references=refs, predictions=hyps)

def compute_bleu(refs, hyps):
    return sacrebleu.compute(predictions=hyps, references=[[r] for r in refs])

