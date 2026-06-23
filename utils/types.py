from dataclasses import dataclass
from typing import Optional

@dataclass
class Result:
    model: str
    audio_file: str
    duration_s: float
    latency_s: float
    wer: float
    cer: float
    word_count: int
    char_count: int
    unique_words: int
    reference: str
    hypothesis: str
    dialogue: str
    error: Optional[str] = None

    @property
    def rtf(self) -> float:
        return self.latency_s / self.duration_s if self.duration_s > 0 else 0.0