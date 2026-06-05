import os, re, time, json, csv, argparse, logging, inspect, importlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

class WhisperXBackend:
    """
    WhisperX – faster-whisper under the hood + word-level forced alignment.
    https://github.com/m-bain/whisperX
    """

    def __init__(
        self,
        model_name:   str = "large-v3",
        device:       str = "cuda",
        compute_type: str = "float16",
        batch_size:   int = 16,
        language:     Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        import whisperx
        log.info(f"Loading WhisperX ({model_name}) on {device} …")
        self.whisperx   = whisperx
        self.model      = whisperx.load_model(
            model_name, device,
            compute_type=compute_type,
            language=language,
        )
        self.device     = device
        self.batch_size = batch_size
        self.language   = language
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.align_cache: dict[str, tuple] = {}
        self.diarizer = None

        DiarizationPipeline = getattr(whisperx, "DiarizationPipeline", None)
        if DiarizationPipeline is None and importlib.util.find_spec("whisperx.diarize") is not None:
            diarize_module = importlib.import_module("whisperx.diarize")
            DiarizationPipeline = getattr(diarize_module, "DiarizationPipeline", None)

        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("PYANNOTE_TOKEN")
        )
        if hf_token and DiarizationPipeline is not None:
            init_params = set(inspect.signature(DiarizationPipeline.__init__).parameters.keys())
            diarizer_kwargs = {}
            if "device" in init_params:
                diarizer_kwargs["device"] = device

            token_arg = next(
                (
                    name for name in ("token", "use_auth_token", "auth_token", "hf_token")
                    if name in init_params
                ),
                None,
            )
            if token_arg is not None:
                diarizer_kwargs[token_arg] = hf_token

            self.diarizer = DiarizationPipeline(**diarizer_kwargs)
            log.info("WhisperX diarization pipeline ready.")
        elif hf_token and DiarizationPipeline is None:
            log.warning("WhisperX diarization class not found in installed whisperx package.")
        else:
            log.warning(
                "No HF token found (HF_TOKEN/HUGGINGFACE_TOKEN/PYANNOTE_TOKEN). "
                "Proceeding without diarization."
            )
        log.info("WhisperX ready.")

    def _load_align_model(self, language_code: str):
        if language_code not in self.align_cache:
            self.align_cache[language_code] = self.whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )
        return self.align_cache[language_code]

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

    def _dialogue_from_word_segments(self, diarized_result: dict) -> str:
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
        lines = self._normalize_speaker_labels(lines)
        return "\n".join(lines).strip()

    def transcribe(self, audio_path: str) -> dict[str, str]:
        audio  = self.whisperx.load_audio(audio_path)
        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language,
        )

        language_code = result.get("language") or self.language
        if language_code:
            align_model, metadata = self._load_align_model(language_code)
            result = self.whisperx.align(
                result["segments"], align_model, metadata,
                audio, self.device,
                return_char_alignments=False,
            )

        dialogue = ""
        if self.diarizer is not None:
            diarize_kwargs = {}
            if self.min_speakers is not None:
                diarize_kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                diarize_kwargs["max_speakers"] = self.max_speakers

            diarize_segments = self.diarizer(audio, **diarize_kwargs)
            diarized = self.whisperx.assign_word_speakers(
                diarize_segments,
                result,
            )
            dialogue = self._dialogue_from_word_segments(diarized)
            if not dialogue:
                dialogue = self._to_dialogue(diarized.get("segments", []))
                if dialogue:
                    dialogue = "\n".join(self._normalize_speaker_labels(dialogue.splitlines()))
            speakers = {
                (s.get("speaker") or "")
                for s in diarized.get("segments", [])
                if (s.get("text") or "").strip()
            }
            log.info(
                f"WhisperX diarization speakers for {Path(audio_path).name}: {len([s for s in speakers if s])}"
            )

        plain = " ".join(seg.get("text", "").strip() for seg in result.get("segments", []))
        if not dialogue:
            dialogue = f"SPEAKER_00: {plain.strip()}" if plain.strip() else ""

        return {"text": plain, "dialogue": dialogue}
