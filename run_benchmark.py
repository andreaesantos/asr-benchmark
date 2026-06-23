# run_benchmark.py
import argparse
import importlib
import logging
from pathlib import Path

from utils.loaders import load_samples
from utils.metrics import compute_wer, compute_cer
from utils.plots import plot_benchmark_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BACKEND_REGISTRY = {
    "whisperx": "backends.whisperx.WhisperXBackend",
    "omni": "backends.omni.OmniBackend",
    "vibevoice": "backends.vibevoice.VibeVoiceBackend",
}

def load_backend(name: str, **kwargs):
    module_path, cls_name = BACKEND_REGISTRY[name].rsplit(".", 1)
    module = importlib.import_module(module_path)   # only imports THIS backend's deps
    cls = getattr(module, cls_name)
    return cls(**kwargs)

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filter", required=True, choices=BACKEND_REGISTRY.keys(),
                         help="Which model to run (one per process/env).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    samples = load_samples(audio_dir="...", transcript_dir="...")

    model = load_backend(args.model_filter, device=args.device)   # <-- called here
    run_benchmark(model, samples)