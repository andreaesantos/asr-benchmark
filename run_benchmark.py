# run_benchmark.py
import logging
from backends.whisperx import WhisperXBackend
from backends.omni import OmniBackend
from utils.loaders import load_samples
from utils.metrics import compute_wer, compute_cer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

def run_benchmark(model, samples):
    results = []
    for sample in samples:
        try:
            prediction = model.transcribe(sample.audio_path)
            # Standardized evaluation
            wer = compute_wer(sample.reference, prediction['text'])
            cer = compute_cer(sample.reference, prediction['text'])
            results.append({"model": model.name, "wer": wer, "cer": cer})
        except Exception as e:
            log.error(f"Error processing {sample.audio_path}: {e}")
            results.append({"model": model.name, "wer": None, "cer": None, "error": str(e)})
    return results