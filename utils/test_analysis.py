from utils.analysis import compute_analysis, _agreement_ratio
import pytest
from utils.text import word_tokens
from utils.types import Result

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