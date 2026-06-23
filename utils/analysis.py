import csv, logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

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
