from pathlib import Path

# Use .expanduser() to resolve the ~ and convert the string to a Path object
ROOT = Path("~/Projects/asr-benchmark").expanduser()

# Now these are Path objects, so the / operator will work perfectly
DATA = ROOT / "data"
RESULTS = ROOT / "benchmark_results"

# Ensure directories exist
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)