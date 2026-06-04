# run_benchmark.py
from backends.whisperx import WhisperXBackend
from backends.omni import OmniBackend
from utils.loaders import load_samples
from utils.metrics import compute_wer, compute_cer

def run_benchmark(model, samples):
    results = []
    for sample in samples:
        prediction = model.transcribe(sample.audio_path)
        # Standardized evaluation
        wer = compute_wer(sample.reference, prediction['text'])
        results.append({"model": model.name, "wer": wer})
    return results