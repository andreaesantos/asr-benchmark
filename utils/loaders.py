import json
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
import logging
from utils.utils import normalise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

@dataclass
class Sample:
    audio_path: str
    reference: str  # Normalized ground truth
    duration_s: float        # <--- Add this line

def load_and_normalize_transcript(path: Path) -> str:
    """Adapts your specific JSON/TXT formats to a simple string."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        # Handles both list-of-dicts and flat-dict structures
        if isinstance(data, list):
            return " ".join(item.get("Content", "") for item in data)
        return data.get("Content", "")
    return path.read_text()

# ──────────────────────────────────────────────────────────────────────────────
# Audio / sample loading
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _audio_duration(audio_path: Path) -> float:
    try:
        import soundfile as sf
        return sf.info(str(audio_path)).duration
    except Exception:
        pass
    try:
        import librosa
        return librosa.get_duration(path=str(audio_path))
    except Exception:
        return 0.0


def _parse_transcript_file(path: Path) -> Optional[str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("text") or data.get("transcript", "")
        if isinstance(data, list):
            return " ".join(
                seg.get("text", "") for seg in data if isinstance(seg, dict)
            )
        return None
    return raw   # plain .txt


def load_samples(
    audio_dir: str,
    transcript_dir: Optional[str] = None,
) -> list[Sample]:
    """
    Load audio files.  If `transcript_dir` is given, pair each audio file with
    its ground-truth transcript (skips files with no match).  Without
    `transcript_dir`, every audio file becomes a sample with reference=None.
    """
    audio_dir_path = Path(audio_dir)
    trans_path     = Path(transcript_dir) if transcript_dir else None
    samples: list[Sample] = []

    for audio_path in sorted(audio_dir_path.iterdir()):
        if audio_path.suffix.lower() not in AUDIO_EXTS:
            continue

        reference: Optional[str] = None

        if trans_path is not None:
            ref_raw: Optional[str] = None
            for ext in (".txt", ".json"):
                candidate = trans_path / (audio_path.stem + ext)
                if candidate.exists():
                    ref_raw = _parse_transcript_file(candidate)
                    break
            if ref_raw is None:
                log.warning(f"No transcript for {audio_path.name} — skipping.")
                continue
            reference = normalise(ref_raw)

        duration_s = _audio_duration(audio_path)

        samples.append(Sample(
            audio_path = str(audio_path),
            duration_s = duration_s,
            reference  = reference,
        ))

    mode = "benchmark" if trans_path else "analysis"
    log.info(f"Loaded {len(samples)} sample(s) from '{audio_dir}' [{mode} mode].")
    return samples
