#!/usr/bin/env python3
import argparse
import gc
from pathlib import Path
import torch
from tqdm import tqdm
import whisperx
from whisperx.diarize import DiarizationPipeline


def find_audio_files(root_dir: Path) -> list:
    """Recursively find all .wav files in a directory."""
    audio_files = sorted(root_dir.rglob("*.wav"))
    return [f for f in audio_files if not f.name.startswith("._")]


def filter_untranscribed(audio_files: list) -> list:
    """Keep only files that don't have a transcript yet."""
    untranscribed = []
    for audio_file in audio_files:
        output_txt = audio_file.with_stem(
            f"{audio_file.stem}_transcript_whisperx"
        ).with_suffix(".txt")
        if not output_txt.exists():
            untranscribed.append(audio_file)
    return untranscribed


def process_audio_file(
    audio_file: Path,
    model,
    diarize_model,
    device: str,
    use_diarization: bool,
    batch_size: int,
    language: str,
):
    """Process a single audio file through ASR pipeline."""
    try:
        print(f"\nProcessing: {audio_file.name}")

        audio = whisperx.load_audio(str(audio_file))
        result = model.transcribe(audio, batch_size=batch_size, language=language)

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        del model_a, metadata
        torch.cuda.empty_cache()
        gc.collect()

        if use_diarization and diarize_model is not None:
            print("  (Running diarization...)")
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return result, None
    except Exception as e:
        print(f"  ✗ Error processing {audio_file.name}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return None, str(e)


def fmt(t):
    h, m, s = int(t // 3600), int((t % 3600) // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def save_transcript(result, audio_file: Path):
    output_txt = audio_file.with_stem(
        f"{audio_file.stem}_transcript_whisperx"
    ).with_suffix(".txt")
    lines = []
    for seg in result["segments"]:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker") or "SPEAKER_00"
        lines.append(
            f"[{fmt(seg.get('start', 0))} --> {fmt(seg.get('end', 0))}] {speaker}: {text}"
        )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_txt


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX Batch Transcription CLI Tool"
    )

    # Required Position Argument
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Path to the directory containing audio files (e.g., ~/Projects/data/)",
    )

    # Optional Arguments
    parser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2"],
        help="Whisper model size to use (default: large-v2) !!! Change if you have limited GPU memory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="fr",
        help="Language code for transcription (default: fr)",
    )
    parser.add_argument(
        "--diarize", action="store_true", help="Enable speaker diarization"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers for diarization if known (default: 2)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for the diarization pipeline",
    )

    args = parser.parse_args()

    # Resolve paths 
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not audio_dir.exists() or not audio_dir.is_dir():
        print(f"Error: Directory does not exist: {audio_dir}")
        return

    # Automatically determine compute_type based on device
    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float16" if args.device == "cuda" else "int8"

    # Filter pipeline
    audio_files = filter_untranscribed(find_audio_files(audio_dir))
    print(f"Found {len(audio_files)} files to process in {audio_dir}.")

    if not audio_files:
        print("Everything is already transcribed!")
        return

    # Initialization
    print(
        f"Initializing WhisperX model '{args.model}' on {args.device} ({compute_type})..."
    )
    model = whisperx.load_model(args.model, args.device, compute_type=compute_type)

    diarize_model = None
    if args.diarize:
        print("Initializing diarization model...")
        diarize_model = DiarizationPipeline(
            token=args.hf_token,
            device=args.device,
            num_speakers=args.num_speakers,
            return_embeddings=True,
        )

    # Loop execution
    results = []
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        result, error = process_audio_file(
            audio_file=audio_file,
            model=model,
            diarize_model=diarize_model,
            device=args.device,
            use_diarization=args.diarize,
            batch_size=args.batch_size,
            language=args.language,
        )

        if result:
            save_transcript(result, audio_file)
            results.append({"status": "success"})
        else:
            results.append({"status": "failed"})

        torch.cuda.empty_cache()
        gc.collect()

    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"\nProcessing complete. Succeeded: {succeeded}/{len(audio_files)}")


if __name__ == "__main__":
    main()