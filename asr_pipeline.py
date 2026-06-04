"""
ASR Pipeline
============
Runs WhisperX and/or Qwen2-Audio-Omni against audio files.

Two modes
---------
  BENCHMARK mode  (--transcript_dir supplied)
      Compares both models against ground-truth transcripts.
      Metrics : WER, CER, latency, RTF (real-time factor)
      Output  : results/summary.csv + results/per_file.csv + console table

  ANALYSIS mode   (no --transcript_dir)
      Transcribes audio with both models and produces a rich
      cross-model analysis (word counts, vocabulary, agreement, …).
      Output  : results/analysis.csv + results/analysis_summary.csv
              + console report

Usage – benchmark:
    python asr_pipeline.py \\
        --audio_dir data/audio \\
        --transcript_dir data/transcripts

Usage – analysis only:
    python asr_pipeline.py \\
        --audio_dir data/audio

Common options:
    --models whisperx omni          # default: both
    --device cuda                   # or cpu
    --whisperx_model large-v3
    --omni_4bit                     # 4-bit quant for Omni (saves VRAM)
    --language en                   # force language code (auto-detect if omitted)
    --output_dir results
"""

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

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    audio_path: str
    duration_s: float = 0.0
    reference:  Optional[str] = None   # normalised ground-truth; None in analysis mode


@dataclass
class Result:
    """One model × one audio file."""
    model:       str
    audio_file:  str
    hypothesis:  str
    latency_s:   float
    duration_s:  float
    dialogue:    str = ""
    # benchmark fields (populated only when reference is available)
    reference:   str   = ""
    wer:         float = -1.0
    cer:         float = -1.0
    # analysis fields (always populated)
    word_count:  int   = 0
    char_count:  int   = 0
    unique_words:int   = 0
    # errors
    error:       str   = ""

    @property
    def rtf(self) -> float:
        """Real-Time Factor = latency / audio_duration  (lower is faster)."""
        return round(self.latency_s / self.duration_s, 4) if self.duration_s else -1.0


# ──────────────────────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation (keep apostrophes), collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_tokens(text: str) -> list[str]:
    return text.split() if text else []


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


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_wer(ref: str, hyp: str) -> float:
    import jiwer
    return jiwer.wer(ref, hyp)


def compute_cer(ref: str, hyp: str) -> float:
    import jiwer
    return jiwer.cer(ref, hyp)


# ──────────────────────────────────────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────────────────────────────────────

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


class OmniBackend:
    """
    Qwen2-Audio-Omni (Qwen2-Audio-7B-Instruct variant with Omni capabilities).
    Falls back gracefully to the standard Qwen2-Audio-7B-Instruct if the Omni
    checkpoint is unavailable.

    HuggingFace: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
    Omni:        https://huggingface.co/Qwen/Qwen2.5-Omni-7B  (when released)

    Requires: pip install transformers>=4.45 accelerate soundfile librosa
    GPU with ≥16 GB VRAM recommended (or use load_in_4bit=True).
    """

    # Prefer the Omni checkpoint; fall back to the standard instruct model.
    DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"
    FALLBACK_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
    PROMPT = "Please transcribe the spoken content in this audio exactly."

    def __init__(
        self,
        model_name:   str = DEFAULT_MODEL,
        device:       str = "cuda",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoProcessor

        log.info(f"Loading Omni model ({model_name}) …")

        self.processor = self._load_processor(model_name)
        self.model     = self._load_model(model_name, load_in_4bit, torch)
        self.device    = device
        log.info("Omni ready.")

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_processor(model_name: str):
        from transformers import AutoProcessor
        try:
            return AutoProcessor.from_pretrained(model_name)
        except Exception as exc:
            log.warning(
                f"Could not load processor for '{model_name}': {exc}\n"
                f"  → Falling back to {OmniBackend.FALLBACK_MODEL}"
            )
            return AutoProcessor.from_pretrained(OmniBackend.FALLBACK_MODEL)

    @staticmethod
    def _load_model(model_name: str, load_in_4bit: bool, torch):
        kwargs: dict = {"device_map": "auto"}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        # Try Omni-specific class first, then generic conditional generation.
        for cls_name in (
            "Qwen2_5OmniForConditionalGeneration",
            "Qwen2AudioForConditionalGeneration",
        ):
            try:
                import transformers
                cls = getattr(transformers, cls_name)
                return cls.from_pretrained(model_name, **kwargs)
            except (AttributeError, OSError, Exception):
                continue

        raise RuntimeError(
            f"Could not load any compatible model class for '{model_name}'. "
            "Ensure transformers>=4.45 is installed and the checkpoint exists."
        )

    # ── public API ───────────────────────────────────────────────────────────

    def _build_inputs(self, text_prompt: str, audio):
        """Build processor inputs robustly across transformers/Qwen variants."""
        audio_keys = ("audio", "audios", "input_audio", "speech")
        feature_keys = {"input_features", "audio_values", "feature_attention_mask"}

        for key in audio_keys:
            try:
                inputs = self.processor(
                    text=text_prompt,
                    **{key: [audio]},
                    return_tensors="pt",
                )
                if any(k in inputs for k in feature_keys):
                    return inputs
            except Exception:
                continue

        # Last resort: text-only (may still produce output, but not true ASR)
        log.warning(
            "Omni processor did not accept audio input keys; using text-only fallback."
        )
        return self.processor(text=text_prompt, return_tensors="pt")

    def transcribe(self, audio_path: str) -> dict[str, str]:
        import librosa, torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text",  "text": self.PROMPT},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        sr        = self.processor.feature_extractor.sampling_rate
        audio, _  = librosa.load(audio_path, sr=sr, mono=True)

        inputs = self._build_inputs(text_prompt, audio)

        # With device_map="auto", model.device can be "meta". Don't force inputs there.
        model_device = getattr(self.model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            try:
                inputs = inputs.to(model_device)
            except Exception:
                pass

        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=512)

        # Normalize generate outputs across model variants.
        if isinstance(generated, tuple):
            generated_ids = generated[0]
        elif hasattr(generated, "sequences"):
            generated_ids = generated.sequences
        else:
            generated_ids = generated

        # Strip prompt tokens when possible.
        prompt_len = 0
        if isinstance(inputs, dict) and "input_ids" in inputs:
            prompt_len = int(inputs["input_ids"].size(1))

        if prompt_len and getattr(generated_ids, "ndim", 0) == 2 and generated_ids.size(1) > prompt_len:
            generated_ids = generated_ids[:, prompt_len:]

        plain = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return {
            "text": plain,
            "dialogue": f"SPEAKER_00: {plain}" if plain else "",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core transcription loop
# ──────────────────────────────────────────────────────────────────────────────

def _transcribe_all(
    samples:  list[Sample],
    backends: dict,
) -> list[Result]:
    """Run every backend over every sample; return raw Result list."""
    all_results: list[Result] = []

    for model_name, backend in backends.items():
        log.info(f"\n{'─'*60}\nRunning model: {model_name}\n{'─'*60}")

        for sample in samples:
            audio_file = Path(sample.audio_path).name
            log.info(f"  Transcribing {audio_file} …")

            try:
                t0  = time.perf_counter()
                tr  = backend.transcribe(sample.audio_path)
                lat = time.perf_counter() - t0
                if isinstance(tr, dict):
                    hyp = tr.get("text", "")
                    dialogue = tr.get("dialogue", "")
                else:
                    hyp = str(tr)
                    dialogue = hyp
                hyp_norm = normalise(hyp)
                tokens   = word_tokens(hyp_norm)

                result = Result(
                    model      = model_name,
                    audio_file = audio_file,
                    hypothesis = hyp_norm,
                    dialogue   = dialogue,
                    latency_s  = round(lat, 3),
                    duration_s = sample.duration_s,
                    word_count = len(tokens),
                    char_count = len(hyp_norm),
                    unique_words = len(set(tokens)),
                )

                if sample.reference is not None:
                    wer = compute_wer(sample.reference, hyp_norm)
                    cer = compute_cer(sample.reference, hyp_norm)
                    result.reference = sample.reference
                    result.wer       = round(wer, 4)
                    result.cer       = round(cer, 4)
                    log.info(
                        f"    WER={wer:.2%}  CER={cer:.2%}  "
                        f"lat={lat:.2f}s  RTF={result.rtf:.3f}"
                    )
                else:
                    log.info(
                        f"    words={result.word_count}  "
                        f"lat={lat:.2f}s  RTF={result.rtf:.3f}"
                    )

                all_results.append(result)

            except Exception as exc:
                log.error(f"    FAILED: {exc}")
                all_results.append(Result(
                    model      = model_name,
                    audio_file = audio_file,
                    hypothesis = "",
                    dialogue   = "",
                    latency_s  = 0.0,
                    duration_s = sample.duration_s,
                    reference  = sample.reference or "",
                    wer        = 1.0 if sample.reference else -1.0,
                    cer        = 1.0 if sample.reference else -1.0,
                    error      = str(exc),
                ))

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK mode
# ──────────────────────────────────────────────────────────────────────────────

def compute_benchmark_summary(results: list[Result]) -> list[dict]:
    buckets: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        buckets[r.model].append(r)

    rows = []
    for model, rs in buckets.items():
        valid = [r for r in rs if not r.error]
        n     = len(valid) or 1
        total_audio = sum(r.duration_s for r in valid)
        total_lat   = sum(r.latency_s  for r in valid)
        rows.append({
            "model":            model,
            "num_files":        len(rs),
            "avg_wer":          round(sum(r.wer for r in valid) / n, 4),
            "avg_cer":          round(sum(r.cer for r in valid) / n, 4),
            "avg_latency_s":    round(sum(r.latency_s for r in valid) / n, 3),
            "overall_rtf":      round(total_lat / total_audio, 4) if total_audio else -1.0,
            "avg_word_count":   round(sum(r.word_count for r in valid) / n, 1),
            "errors":           len(rs) - len(valid),
        })
    return rows


def print_benchmark_table(summary: list[dict]) -> None:
    header = (
        f"{'Model':<20} {'Files':>6} {'Avg WER':>9} {'Avg CER':>9} "
        f"{'Avg Lat(s)':>11} {'RTF':>7} {'Errors':>7}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for row in summary:
        rtf_str = f"{row['overall_rtf']:.3f}" if row['overall_rtf'] >= 0 else "  n/a"
        print(
            f"{row['model']:<20} {row['num_files']:>6} "
            f"{row['avg_wer']:>8.2%} {row['avg_cer']:>8.2%} "
            f"{row['avg_latency_s']:>10.2f}s "
            f"{rtf_str:>7} {row['errors']:>7}"
        )
    print(sep)


def save_benchmark_csvs(results: list[Result], out_path: Path) -> None:
    if not results:
        log.warning("No results to save.")
        return

    # Per-file CSV (all fields)
    per_file_csv = out_path / "per_file.csv"
    fieldnames   = [
        "model", "audio_file", "duration_s", "latency_s", "rtf",
        "wer", "cer", "word_count", "char_count", "unique_words",
        "reference", "hypothesis", "dialogue", "error",
    ]
    with open(per_file_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["rtf"] = r.rtf
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    log.info(f"Per-file results  → {per_file_csv}")

    # Summary CSV
    summary     = compute_benchmark_summary(results)
    summary_csv = out_path / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    log.info(f"Summary results   → {summary_csv}")

    print_benchmark_table(summary)


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS mode
# ──────────────────────────────────────────────────────────────────────────────

def _agreement_ratio(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Jaccard similarity on unigram bags-of-words.
    Ranges 0 (no overlap) → 1 (identical vocabulary).
    """
    set_a, set_b = set(tokens_a), set(tokens_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return round(len(set_a & set_b) / len(union), 4)


def compute_analysis(results: list[Result]) -> dict:
    """
    Cross-model analysis for a set of Result objects (no ground truth).

    Returns a dict with:
      per_file   – list[dict]  (one row per audio × model)
      summary    – list[dict]  (one row per model)
      comparison – list[dict]  (one row per audio file, cross-model stats)
    """
    # ── per-file rows ────────────────────────────────────────────────────────
    per_file = []
    for r in results:
        tokens = word_tokens(r.hypothesis)
        word_freq = Counter(tokens)
        top5 = ", ".join(f"{w}({c})" for w, c in word_freq.most_common(5))
        per_file.append({
            "model":        r.model,
            "audio_file":   r.audio_file,
            "duration_s":   r.duration_s,
            "latency_s":    r.latency_s,
            "rtf":          r.rtf,
            "word_count":   r.word_count,
            "words_per_min":round(r.word_count / (r.duration_s / 60), 1)
                            if r.duration_s else -1,
            "char_count":   r.char_count,
            "unique_words": r.unique_words,
            "type_token_ratio": round(r.unique_words / r.word_count, 4)
                                 if r.word_count else 0.0,
            "top5_words":   top5,
            "hypothesis":   r.hypothesis,
            "dialogue":     r.dialogue,
            "error":        r.error,
        })

    # ── per-model summary ────────────────────────────────────────────────────
    buckets: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        buckets[r.model].append(r)

    summary = []
    for model, rs in buckets.items():
        valid = [r for r in rs if not r.error]
        n     = len(valid) or 1
        total_audio = sum(r.duration_s for r in valid)
        total_lat   = sum(r.latency_s  for r in valid)
        all_tokens  = [t for r in valid for t in word_tokens(r.hypothesis)]
        top5_global = ", ".join(
            f"{w}({c})" for w, c in Counter(all_tokens).most_common(5)
        )
        summary.append({
            "model":            model,
            "num_files":        len(rs),
            "errors":           len(rs) - len(valid),
            "avg_word_count":   round(sum(r.word_count for r in valid) / n, 1),
            "avg_words_per_min":round(
                sum(r.word_count / (r.duration_s / 60)
                    for r in valid if r.duration_s) / n, 1
            ) if valid else -1,
            "avg_unique_words": round(sum(r.unique_words for r in valid) / n, 1),
            "avg_type_token_ratio": round(
                sum(r.unique_words / r.word_count
                    for r in valid if r.word_count) / n, 4
            ) if valid else 0.0,
            "avg_latency_s":    round(total_lat / n, 3),
            "overall_rtf":      round(total_lat / total_audio, 4)
                                if total_audio else -1.0,
            "global_top5_words": top5_global,
        })

    # ── cross-model comparison (one row per audio file) ──────────────────────
    # Index results by audio_file → model
    file_model: dict[str, dict[str, Result]] = defaultdict(dict)
    for r in results:
        file_model[r.audio_file][r.model] = r

    model_names = list(buckets.keys())
    comparison  = []
    if len(model_names) >= 2:
        m0, m1 = model_names[0], model_names[1]
        for audio_file, by_model in file_model.items():
            r0 = by_model.get(m0)
            r1 = by_model.get(m1)
            if r0 and r1 and not r0.error and not r1.error:
                t0 = word_tokens(r0.hypothesis)
                t1 = word_tokens(r1.hypothesis)
                agreement = _agreement_ratio(t0, t1)
                shared = sorted(set(t0) & set(t1))[:10]
                only_m0 = sorted(set(t0) - set(t1))[:10]
                only_m1 = sorted(set(t1) - set(t0))[:10]
                comparison.append({
                    "audio_file":         audio_file,
                    f"{m0}_word_count":   r0.word_count,
                    f"{m1}_word_count":   r1.word_count,
                    "word_count_delta":   r0.word_count - r1.word_count,
                    f"{m0}_words_per_min":
                        round(r0.word_count / (r0.duration_s / 60), 1)
                        if r0.duration_s else -1,
                    f"{m1}_words_per_min":
                        round(r1.word_count / (r1.duration_s / 60), 1)
                        if r1.duration_s else -1,
                    "vocabulary_agreement": agreement,
                    "shared_words_sample":  " | ".join(shared),
                    f"only_in_{m0}_sample": " | ".join(only_m0),
                    f"only_in_{m1}_sample": " | ".join(only_m1),
                    f"{m0}_latency_s":     r0.latency_s,
                    f"{m1}_latency_s":     r1.latency_s,
                })

    return {"per_file": per_file, "summary": summary, "comparison": comparison}


def print_analysis_report(analysis: dict) -> None:
    summary    = analysis["summary"]
    comparison = analysis["comparison"]

    # ── per-model summary table ──────────────────────────────────────────────
    header = (
        f"{'Model':<20} {'Files':>6} {'AvgWords':>9} "
        f"{'WPM':>7} {'TTR':>7} {'RTF':>7} {'Errors':>7}"
    )
    sep = "─" * len(header)
    print(f"\n{'═'*len(header)}\n  ANALYSIS SUMMARY\n{'═'*len(header)}")
    print(f"{header}\n{sep}")
    for row in summary:
        rtf = f"{row['overall_rtf']:.3f}" if row['overall_rtf'] >= 0 else "  n/a"
        wpm = f"{row['avg_words_per_min']:.0f}" if row['avg_words_per_min'] >= 0 else " n/a"
        print(
            f"{row['model']:<20} {row['num_files']:>6} "
            f"{row['avg_word_count']:>9.1f} "
            f"{wpm:>7} "
            f"{row['avg_type_token_ratio']:>7.4f} "
            f"{rtf:>7} "
            f"{row['errors']:>7}"
        )
    print(sep)
    for row in summary:
        print(f"  {row['model']} top-5 words: {row['global_top5_words']}")

    # ── cross-model comparison ───────────────────────────────────────────────
    if comparison:
        print(f"\n{'═'*len(header)}\n  CROSS-MODEL COMPARISON (per file)\n{'═'*len(header)}")
        for row in comparison:
            print(f"\n  {row['audio_file']}")
            for k, v in row.items():
                if k != "audio_file":
                    print(f"    {k:<35} {v}")
    print(f"\n{'═'*len(header)}\n")


def save_analysis_csvs(analysis: dict, out_path: Path) -> None:
    def _write(rows: list[dict], filename: str) -> None:
        if not rows:
            return
        csv_path = out_path / filename
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"{filename:<30} → {csv_path}")

    _write(analysis["per_file"],   "analysis_per_file.csv")
    _write(analysis["summary"],    "analysis_summary.csv")
    _write(analysis["comparison"], "analysis_comparison.csv")


def save_dialogue_transcripts(results: list[Result], out_path: Path) -> None:
    """Write one dialogue transcript .txt file per audio result."""
    transcripts_dir = out_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    models = sorted({r.model for r in results})
    multi_model = len(models) > 1
    written = 0

    for r in results:
        if r.error or not r.dialogue.strip():
            continue
        stem = Path(r.audio_file).stem
        filename = f"{stem}.{r.model}.txt" if multi_model else f"{stem}.txt"
        out_file = transcripts_dir / filename
        out_file.write_text(r.dialogue.strip() + "\n", encoding="utf-8")
        written += 1

    log.info(f"Dialogue transcript files → {transcripts_dir} ({written} file(s))")


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    samples:   list[Sample],
    backends:  dict,
    out_dir:   str = "results",
) -> list[Result]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = _transcribe_all(samples, backends)

    if not all_results:
        log.error("No results produced.")
        return all_results

    benchmark_mode = any(s.reference is not None for s in samples)

    if benchmark_mode:
        log.info("\n── Benchmark mode: computing WER/CER metrics ──")
        save_benchmark_csvs(all_results, out_path)
    else:
        log.info("\n── Analysis mode: computing cross-model statistics ──")
        analysis = compute_analysis(all_results)
        save_analysis_csvs(analysis, out_path)
        print_analysis_report(analysis)

    save_dialogue_transcripts(all_results, out_path)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ASR Pipeline: WhisperX vs Qwen2-Audio-Omni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--audio_dir", required=True,
        help="Directory containing audio files (.wav / .mp3 / .flac / …)",
    )
    p.add_argument(
        "--transcript_dir", default=None,
        help="[Optional] Ground-truth .txt / .json files — activates benchmark mode",
    )
    p.add_argument(
        "--models", nargs="+", default=["whisperx", "omni"],
        choices=["whisperx", "omni"],
        help="Which models to run",
    )
    p.add_argument("--output_dir", default="results")
    p.add_argument("--device",     default="cuda",
                   help="Torch device: cuda | cpu")
    p.add_argument("--language",   default=None,
                   help="Force a BCP-47 language code, e.g. 'en' (default: auto-detect)")

    # WhisperX
    wx = p.add_argument_group("WhisperX options")
    wx.add_argument("--whisperx_model",        default="large-v3",
                    choices=["tiny","base","small","medium",
                             "large-v2","large-v3"],
                    help="Model size")
    wx.add_argument("--whisperx_compute_type", default="float16",
                    choices=["float16","float32","int8"])
    wx.add_argument("--whisperx_batch_size",   type=int, default=16)
    wx.add_argument("--whisperx_min_speakers", type=int, default=0,
                    help="Minimum number of speakers for diarization (set 0 to disable constraint)")
    wx.add_argument("--whisperx_max_speakers", type=int, default=0,
                    help="Maximum number of speakers for diarization (set 0 to disable constraint)")

    # Omni
    om = p.add_argument_group("Omni options")
    om.add_argument("--omni_model",  default=OmniBackend.DEFAULT_MODEL,
                    help="HuggingFace model ID for Omni")
    om.add_argument("--omni_4bit",   action="store_true",
                    help="Load in 4-bit quantisation (saves VRAM)")

    return p.parse_args()


def main():
    args = parse_args()

    samples = load_samples(args.audio_dir, args.transcript_dir)
    if not samples:
        log.error("No samples found — check your audio (and transcript) directories.")
        return

    backends: dict = {}

    if "whisperx" in args.models:
        min_spk = args.whisperx_min_speakers if args.whisperx_min_speakers > 0 else None
        max_spk = args.whisperx_max_speakers if args.whisperx_max_speakers > 0 else None
        backends["whisperx"] = WhisperXBackend(
            model_name   = args.whisperx_model,
            device       = args.device,
            compute_type = args.whisperx_compute_type,
            batch_size   = args.whisperx_batch_size,
            language     = args.language,
            min_speakers = min_spk,
            max_speakers = max_spk,
        )

    if "omni" in args.models:
        backends["omni"] = OmniBackend(
            model_name   = args.omni_model,
            device       = args.device,
            load_in_4bit = args.omni_4bit,
        )

    run_pipeline(samples, backends, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
