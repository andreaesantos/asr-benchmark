from pathlib import Path

ROOT = Path.home() / "Projects"

DATA = ROOT / "data" / "asr-benchmark"

RESULTS = DATA / "benchmark_results"

# Ensure directories exist
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, 
              exist_ok=True)

