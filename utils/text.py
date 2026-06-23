import re

def normalise(text: str) -> str:
    """Lowercase, strip punctuation (keep apostrophes), collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_tokens(text: str) -> list[str]:
    return text.split() if text else []

def format_timestamped_dialogue(segments: list[dict]) -> str:
    """
    Normalizes diverse ASR outputs into: 
    [start---> end][speaker][content]
    """
    lines = []
    for seg in segments:
        start = seg.get("Start") or seg.get("start") or 0.0
        end = seg.get("End") or seg.get("end") or 0.0
        speaker = seg.get("Speaker") or seg.get("speaker") or "0"
        content = seg.get("Content") or seg.get("text") or ""
        
        # Ensure speaker is normalized (e.g., 0 -> SPEAKER_00)
        spk_label = f"SPEAKER_{int(speaker):02d}"
        
        lines.append(f"[{start:.2f}---> {end:.2f}][{spk_label}][{content.strip()}]")
        
    return "\n".join(lines)
