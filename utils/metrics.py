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
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_wer(ref: str, hyp: str) -> float:
    import jiwer
    return jiwer.wer(ref, hyp)


def compute_cer(ref: str, hyp: str) -> float:
    import jiwer
    return jiwer.cer(ref, hyp)


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
