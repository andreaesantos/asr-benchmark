"""
test_pipeline.py
================
Unit tests for the non-model parts of asr_pipeline.py:
  - text normalisation
  - sample loading  (benchmark mode + analysis mode, txt + json formats)
  - WER / CER computation
  - analysis metrics (word stats, agreement ratio, compute_analysis)
  - benchmark summary aggregation
  - RTF computation

Run with:  python -m pytest test_pipeline.py -v
"""

import json
import struct
import wave
from pathlib import Path
from collections import Counter

import pytest

from asr_pipeline import (
    # text
    normalise,
    word_tokens,
    # I/O
    load_samples,
    # metrics
    compute_wer,
    compute_cer,
    # benchmark
    compute_benchmark_summary,
    # analysis
    compute_analysis,
    _agreement_ratio,
    # data classes
    Sample,
    Result,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_wav(path: str, duration_s: float = 1.0, sample_rate: int = 16_000) -> None:
    """Write a minimal silent WAV file."""
    n = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


def _setup_dirs(tmp_path: Path):
    audio = tmp_path / "audio"
    trans = tmp_path / "transcripts"
    audio.mkdir()
    trans.mkdir()
    return audio, trans


def _make_result(
    model="whisperx",
    audio_file="f.wav",
    hypothesis="hello world",
    wer=0.0,
    cer=0.0,
    latency_s=1.0,
    duration_s=5.0,
    error="",
) -> Result:
    tokens = word_tokens(hypothesis)
    return Result(
        model=model,
        audio_file=audio_file,
        hypothesis=hypothesis,
        latency_s=latency_s,
        duration_s=duration_s,
        reference="hello world",
        wer=wer,
        cer=cer,
        word_count=len(tokens),
        char_count=len(hypothesis),
        unique_words=len(set(tokens)),
        error=error,
    )


# ──────────────────────────────────────────────────────────────────────────────
# normalise
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercase(self):
        assert normalise("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert normalise("Hello, world!") == "hello world"

    def test_keeps_apostrophe(self):
        assert normalise("it's fine") == "it's fine"

    def test_collapses_whitespace(self):
        assert normalise("  too   many   spaces  ") == "too many spaces"

    def test_empty(self):
        assert normalise("") == ""


# ──────────────────────────────────────────────────────────────────────────────
# word_tokens
# ──────────────────────────────────────────────────────────────────────────────

class TestWordTokens:
    def test_splits_on_space(self):
        assert word_tokens("hello world") == ["hello", "world"]

    def test_empty_string(self):
        assert word_tokens("") == []

    def test_single_word(self):
        assert word_tokens("hi") == ["hi"]


# ──────────────────────────────────────────────────────────────────────────────
# load_samples
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadSamplesBenchmarkMode:
    """With transcript_dir supplied → benchmark mode (reference populated)."""

    def test_loads_txt_transcript(self, tmp_path):
        audio, trans = _setup_dirs(tmp_path)
        make_wav(str(audio / "utt1.wav"))
        (trans / "utt1.txt").write_text("Hello world", encoding="utf-8")

        samples = load_samples(str(audio), str(trans))
        assert len(samples) == 1
        assert samples[0].reference == "hello world"

    def test_loads_json_dict_transcript(self, tmp_path):
        audio, trans = _setup_dirs(tmp_path)
        make_wav(str(audio / "utt2.wav"))
        (trans / "utt2.json").write_text(
            json.dumps({"text": "Test sentence."}), encoding="utf-8"
        )
        samples = load_samples(str(audio), str(trans))
        assert samples[0].reference == "test sentence"

    def test_loads_json_segment_list(self, tmp_path):
        audio, trans = _setup_dirs(tmp_path)
        make_wav(str(audio / "utt3.wav"))
        (trans / "utt3.json").write_text(
            json.dumps([{"text": " Hello"}, {"text": " world"}]), encoding="utf-8"
        )
        samples = load_samples(str(audio), str(trans))
        assert samples[0].reference == "hello world"

    def test_skips_audio_without_transcript(self, tmp_path):
        audio, trans = _setup_dirs(tmp_path)
        make_wav(str(audio / "orphan.wav"))
        assert load_samples(str(audio), str(trans)) == []

    def test_skips_non_audio_files(self, tmp_path):
        audio, trans = _setup_dirs(tmp_path)
        (audio / "readme.txt").write_text("ignore me")
        assert load_samples(str(audio), str(trans)) == []

    def test_multiple_samples(self, tmp_path):
        audio, trans = _setup_dirs(tmp_path)
        for i in range(3):
            make_wav(str(audio / f"utt{i}.wav"))
            (trans / f"utt{i}.txt").write_text(f"sentence {i}")
        assert len(load_samples(str(audio), str(trans))) == 3


class TestLoadSamplesAnalysisMode:
    """Without transcript_dir → analysis mode (reference is None)."""

    def test_loads_all_audio_no_transcript_dir(self, tmp_path):
        audio, _ = _setup_dirs(tmp_path)
        for i in range(3):
            make_wav(str(audio / f"utt{i}.wav"))
        samples = load_samples(str(audio))
        assert len(samples) == 3
        assert all(s.reference is None for s in samples)

    def test_skips_non_audio_files_in_analysis_mode(self, tmp_path):
        audio, _ = _setup_dirs(tmp_path)
        (audio / "notes.txt").write_text("not audio")
        make_wav(str(audio / "real.wav"))
        samples = load_samples(str(audio))
        assert len(samples) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_wer(self):
        assert compute_wer("hello world", "hello world") == pytest.approx(0.0)

    def test_perfect_cer(self):
        assert compute_cer("hello", "hello") == pytest.approx(0.0)

    def test_full_deletion_wer(self):
        assert compute_wer("one two three", "") == pytest.approx(1.0)

    def test_one_substitution_wer(self):
        assert compute_wer("hello world", "hello earth") == pytest.approx(0.5)

    def test_cer_partial(self):
        assert compute_cer("hello", "hXllo") == pytest.approx(0.2)

    def test_wer_returns_float(self):
        assert isinstance(compute_wer("a b", "a c"), float)

    def test_cer_returns_float(self):
        assert isinstance(compute_cer("abc", "aXc"), float)


# ──────────────────────────────────────────────────────────────────────────────
# Result.rtf
# ──────────────────────────────────────────────────────────────────────────────

class TestRTF:
    def test_rtf_calculation(self):
        r = _make_result(latency_s=2.0, duration_s=10.0)
        assert r.rtf == pytest.approx(0.2)

    def test_rtf_zero_duration_returns_minus_one(self):
        r = _make_result(latency_s=1.0, duration_s=0.0)
        assert r.rtf == -1.0

    def test_rtf_faster_than_realtime(self):
        r = _make_result(latency_s=1.0, duration_s=5.0)
        assert r.rtf < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# compute_benchmark_summary
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeBenchmarkSummary:
    def _results(self, model, wers, cers, lats, dur=5.0):
        return [
            _make_result(model=model, audio_file=f"f{i}.wav",
                         wer=w, cer=c, latency_s=l, duration_s=dur)
            for i, (w, c, l) in enumerate(zip(wers, cers, lats))
        ]

    def test_averages_correct(self):
        rs = self._results("whisperx", [0.2, 0.4], [0.1, 0.3], [1.0, 3.0])
        row = compute_benchmark_summary(rs)[0]
        assert row["avg_wer"] == pytest.approx(0.3, abs=1e-4)
        assert row["avg_cer"] == pytest.approx(0.2, abs=1e-4)
        assert row["avg_latency_s"] == pytest.approx(2.0, abs=1e-3)

    def test_rtf_overall(self):
        # total latency = 4s, total audio = 10s → RTF = 0.4
        rs = self._results("whisperx", [0.1, 0.1], [0.0, 0.0], [2.0, 2.0], dur=5.0)
        row = compute_benchmark_summary(rs)[0]
        assert row["overall_rtf"] == pytest.approx(0.4, abs=1e-3)

    def test_multiple_models(self):
        rs = (
            self._results("whisperx", [0.1], [0.05], [0.5]) +
            self._results("omni",     [0.2], [0.10], [2.0])
        )
        by_model = {r["model"]: r for r in compute_benchmark_summary(rs)}
        assert set(by_model.keys()) == {"whisperx", "omni"}
        assert by_model["whisperx"]["avg_wer"] == pytest.approx(0.1, abs=1e-4)
        assert by_model["omni"]["avg_wer"]     == pytest.approx(0.2, abs=1e-4)

    def test_ignores_errored_results(self):
        rs = self._results("whisperx", [0.1, 0.9], [0.05, 0.9], [1.0, 0.0])
        rs[1].error = "CUDA OOM"
        row = compute_benchmark_summary(rs)[0]
        assert row["avg_wer"] == pytest.approx(0.1, abs=1e-4)
        assert row["errors"]  == 1

    def test_num_files_counts_all(self):
        rs = self._results("whisperx", [0.0, 0.0, 0.0], [0.0]*3, [1.0]*3)
        assert compute_benchmark_summary(rs)[0]["num_files"] == 3


# ──────────────────────────────────────────────────────────────────────────────
# _agreement_ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestAgreementRatio:
    def test_identical(self):
        assert _agreement_ratio(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _agreement_ratio(["a", "b"], ["c", "d"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # |intersection| = 1, |union| = 3  → 1/3
        ratio = _agreement_ratio(["a", "b"], ["b", "c"])
        assert ratio == pytest.approx(1 / 3, abs=1e-4)

    def test_empty_both(self):
        assert _agreement_ratio([], []) == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_analysis
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeAnalysis:
    def _make_pair(self, hyp_wx, hyp_omni, audio_file="f.wav", duration_s=6.0):
        """Create one whisperx + one omni result for the same audio file."""
        def _r(model, hyp):
            tokens = word_tokens(hyp)
            return Result(
                model=model, audio_file=audio_file,
                hypothesis=hyp, latency_s=1.0, duration_s=duration_s,
                word_count=len(tokens), char_count=len(hyp),
                unique_words=len(set(tokens)),
            )
        return [_r("whisperx", hyp_wx), _r("omni", hyp_omni)]

    def test_summary_has_both_models(self):
        rs = self._make_pair("hello world", "hello earth")
        analysis = compute_analysis(rs)
        models = {r["model"] for r in analysis["summary"]}
        assert models == {"whisperx", "omni"}

    def test_word_count_in_per_file(self):
        rs = self._make_pair("one two three", "one two")
        analysis = compute_analysis(rs)
        wx_row = next(r for r in analysis["per_file"] if r["model"] == "whisperx")
        assert wx_row["word_count"] == 3

    def test_words_per_min_computed(self):
        # 6 words, 60s audio → 6 wpm
        rs = self._make_pair("a b c d e f", "x y z", duration_s=60.0)
        analysis = compute_analysis(rs)
        wx_row = next(r for r in analysis["per_file"] if r["model"] == "whisperx")
        assert wx_row["words_per_min"] == pytest.approx(6.0, abs=0.1)

    def test_type_token_ratio(self):
        # "a a b" → unique=2, total=3 → TTR=0.6667
        rs = self._make_pair("a a b", "x")
        analysis = compute_analysis(rs)
        wx_row = next(r for r in analysis["per_file"] if r["model"] == "whisperx")
        assert wx_row["type_token_ratio"] == pytest.approx(2 / 3, abs=1e-4)

    def test_comparison_contains_agreement(self):
        rs = self._make_pair("hello world", "hello earth")
        analysis = compute_analysis(rs)
        assert len(analysis["comparison"]) == 1
        comp = analysis["comparison"][0]
        assert "vocabulary_agreement" in comp
        # shared: {"hello"}, union: {"hello","world","earth"} → 1/3
        assert comp["vocabulary_agreement"] == pytest.approx(1 / 3, abs=1e-4)

    def test_comparison_word_count_delta(self):
        rs = self._make_pair("a b c d", "x y")   # wx=4, omni=2 → delta=2
        analysis = compute_analysis(rs)
        comp = analysis["comparison"][0]
        assert comp["word_count_delta"] == 2

    def test_no_comparison_when_single_model(self):
        tokens = word_tokens("hello world")
        rs = [Result(
            model="whisperx", audio_file="f.wav",
            hypothesis="hello world", latency_s=1.0, duration_s=5.0,
            word_count=len(tokens), char_count=11, unique_words=2,
        )]
        analysis = compute_analysis(rs)
        assert analysis["comparison"] == []

    def test_errored_results_excluded_from_summary_averages(self):
        rs = self._make_pair("hello world", "hi there")
        rs[0].error = "boom"
        analysis = compute_analysis(rs)
        wx_summary = next(r for r in analysis["summary"] if r["model"] == "whisperx")
        assert wx_summary["errors"] == 1
        # avg_word_count should be 0 (no valid results) or based only on valid ones
        assert wx_summary["avg_word_count"] == pytest.approx(0.0, abs=1e-4)
