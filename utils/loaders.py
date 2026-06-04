import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Sample:
    audio_path: str
    reference: str  # Normalized ground truth

def load_and_normalize_transcript(path: Path) -> str:
    """Adapts your specific JSON/TXT formats to a simple string."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        # Handles both list-of-dicts and flat-dict structures
        if isinstance(data, list):
            return " ".join(item.get("Content", "") for item in data)
        return data.get("Content", "")
    return path.read_text()