from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autocut_ai.schemas import ClipTranscription, Segment, WordToken


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe one extracted audio file with faster-whisper.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--compute-type", required=True)
    parser.add_argument("--beam-size", required=True, type=int)
    parser.add_argument("--clip-id", required=True, type=int)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--vad-filter", action="store_true")
    return parser


def transcribe_to_json(args: argparse.Namespace) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments, info = model.transcribe(
        args.audio_path,
        language="it",
        word_timestamps=True,
        beam_size=args.beam_size,
        best_of=1,
        vad_filter=args.vad_filter,
        condition_on_previous_text=False,
    )
    built_segments: list[Segment] = []
    for segment_index, segment in enumerate(segments, start=1):
        words = [
            WordToken(
                word=word.word.strip(),
                start=float(word.start),
                end=float(word.end),
                probability=float(word.probability) if word.probability is not None else None,
            )
            for word in segment.words or []
        ]
        built_segments.append(
            Segment(
                id=segment_index,
                clip_id=args.clip_id,
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
                words=words,
            )
        )
    transcription = ClipTranscription(
        clip_id=args.clip_id,
        source_path=args.source_path,
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        segments=built_segments,
    )
    Path(args.output_json).write_text(transcription.model_dump_json(), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        transcribe_to_json(args)
    except ImportError as exc:
        print(f"Il pacchetto faster-whisper non e installato: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
