# run_benchmark.py
import logging
from pathlib import Path

from backends.whisperx import WhisperXBackend
from backends.omni import OmniBackend
from backends.vibevoice import VibeVoiceBackend
from utils.loaders import load_samples
from utils.metrics import compute_wer, compute_cer
from utils.plots import plot_benchmark_results

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
    
    plot_benchmark_results(results, output_dir=Path("benchmark_results"))

    return results 