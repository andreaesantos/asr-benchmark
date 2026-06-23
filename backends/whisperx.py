import os, logging
from typing import Optional
import pydantic
import exca as xk

from backends.base import ASRBackend

import torch
import omegaconf
from omegaconf import ListConfig
torch.serialization.add_safe_globals([ListConfig])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

class WhisperXTask(pydantic.BaseModel):
    model_name: str
    audio_path: str
    device: str
    compute_type: str
    batch_size: int
    language: Optional[str]
    min_speakers: Optional[int]
    max_speakers: Optional[int]

    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    @infra.apply
    def run(self) -> dict[str, str]:
        return WhisperXBackend._execute_inference(
            self.model_name, self.audio_path, self.device, 
            self.compute_type, self.batch_size, self.language, 
            self.min_speakers, self.max_speakers
        )

class WhisperXBackend(ASRBackend):
    def __init__(
        self,
        name: str = "whisperx",
        model_name:   str = "large-v3",
        device:       str = "cuda",
        compute_type: str = "int8",
        batch_size:   int = 16,
        language:     Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.name = name
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        log.info(f"WhisperX backend initialized ({model_name}).")

    @staticmethod
    def _execute_inference(model_name, audio_path, device, compute_type, batch_size, language, min_spk, max_spk) -> dict[str, str]:
        """
        Standalone logic container for the cached Task.
        """
        import whisperx
        import importlib

        # Load Model
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)

        # Alignment
        language_code = result.get("language") or language
        if language_code:
            align_model, metadata = whisperx.load_align_model(language_code, device)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

        # Diarization
        diarizer = WhisperXBackend._get_diarizer(device)
        dialogue = ""
        if diarizer:
            diarize_kwargs = {}
            if min_spk is not None: diarize_kwargs["min_speakers"] = min_spk
            if max_spk is not None: diarize_kwargs["max_speakers"] = max_spk
            
            diarize_segments = diarizer(audio, **diarize_kwargs)
            diarized = whisperx.assign_word_speakers(diarize_segments, result)
            dialogue = WhisperXBackend._dialogue_from_word_segments(diarized)

        plain = " ".join(seg.get("text", "").strip() for seg in result.get("segments", []))
        if not dialogue:
            dialogue = f"SPEAKER_00: {plain.strip()}" if plain.strip() else ""
            
        return {"text": plain, "dialogue": dialogue}

    @staticmethod
    def _get_diarizer(device):
        import whisperx
        DiarizationPipeline = getattr(whisperx, "DiarizationPipeline", None)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("PYANNOTE_TOKEN")
        if hf_token and DiarizationPipeline:
            return DiarizationPipeline(device=device, use_auth_token=hf_token)
        return None

    def transcribe(self, audio_path: str) -> dict[str, str]:
        """Calls the cached Task to perform the work."""
        task = WhisperXTask(
            model_name=self.model_name,
            audio_path=audio_path,
            device=self.device,
            compute_type=self.compute_type,
            batch_size=self.batch_size,
            language=self.language,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )
        return task.run()

    @staticmethod
    def _to_dialogue(segments: list[dict]) -> str:
        lines: list[str] = []
        last_speaker = None
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            speaker = seg.get("speaker") or "SPEAKER_00"
            if speaker == last_speaker and lines:
                lines[-1] = lines[-1] + " " + text
            else:
                lines.append(f"{speaker}: {text}")
                last_speaker = speaker
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_speaker_labels(lines: list[str]) -> list[str]:
        """Remap diarizer labels to stable SPEAKER_00/01/... order per file."""
        mapping: dict[str, str] = {}
        out: list[str] = []
        idx = 0
        for line in lines:
            if ":" not in line:
                out.append(line)
                continue
            spk, text = line.split(":", 1)
            spk = spk.strip()
            if spk not in mapping:
                mapping[spk] = f"SPEAKER_{idx:02d}"
                idx += 1
            out.append(f"{mapping[spk]}: {text.strip()}")
        return out

    @staticmethod
    def _dialogue_from_word_segments(diarized_result: dict) -> str:
        """
        Build dialogue from word-level speaker tags to better capture turn-taking.
        """
        words: list[dict] = []
        for seg in diarized_result.get("segments", []):
            for w in seg.get("words", []) or []:
                if not isinstance(w, dict):
                    continue
                token = (w.get("word") or "").strip()
                if not token:
                    continue
                words.append(w)

        if not words:
            return ""

        turns: list[tuple[str, str]] = []
        current_speaker = None
        current_tokens: list[str] = []
        last_end = None

        # Split turns on speaker change or large pause.
        max_pause_s = 0.9
        for w in words:
            speaker = w.get("speaker") or "SPEAKER_00"
            token = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")

            speaker_change = (current_speaker is not None and speaker != current_speaker)
            pause_break = (
                last_end is not None and isinstance(start, (int, float))
                and (float(start) - float(last_end) > max_pause_s)
            )

            if (speaker_change or pause_break) and current_tokens:
                turns.append((current_speaker or "SPEAKER_00", " ".join(current_tokens).strip()))
                current_tokens = []

            current_speaker = speaker
            current_tokens.append(token)
            if isinstance(end, (int, float)):
                last_end = float(end)

        if current_tokens:
            turns.append((current_speaker or "SPEAKER_00", " ".join(current_tokens).strip()))

        lines = [f"{spk}: {txt}" for spk, txt in turns if txt]
        lines = WhisperXBackend._normalize_speaker_labels(lines)
        return "\n".join(lines).strip()
