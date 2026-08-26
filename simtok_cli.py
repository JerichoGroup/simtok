"""Provide the CLI entrypoint for the SimTok pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so internal imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="simtok",
        description="SimTok — synthetic thermal video dataset pipeline.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- collect ---
    collect_parser = sub.add_parser(
        "collect",
        help="Run Isaac Sim and record video/pose/bbox data.",
    )
    collect_parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to generate (overrides TOML).",
    )
    collect_parser.add_argument(
        "--duration", type=int, default=None,
        help="Video duration in seconds per sample (overrides TOML).",
    )
    collect_parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Dataset root directory (overrides TOML).",
    )
    collect_parser.add_argument(
        "--usd-path", type=str, default=None,
        help="USD scene file path (overrides TOML).",
    )

    # --- noise ---
    noise_parser = sub.add_parser(
        "noise",
        help="Apply thermal noise to recorded videos.",
    )
    noise_parser.add_argument(
        "--input-video-dir", type=Path, default=None,
        help="Directory containing input videos (overrides TOML).",
    )
    noise_parser.add_argument(
        "--output-video-dir", type=Path, default=None,
        help="Directory to write processed videos (overrides TOML).",
    )

    # --- json ---
    json_parser = sub.add_parser(
        "json",
        help="Generate per-frame JSON metadata from pose/bbox pickles.",
    )
    json_parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Dataset root directory (overrides TOML).",
    )

    # --- run-all ---
    run_all_parser = sub.add_parser(
        "run-all",
        help="Run the full pipeline: collect → noise → json.",
    )
    run_all_parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to generate (overrides TOML).",
    )
    run_all_parser.add_argument(
        "--duration", type=int, default=None,
        help="Video duration in seconds per sample (overrides TOML).",
    )
    run_all_parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Dataset root directory (overrides TOML).",
    )
    run_all_parser.add_argument(
        "--usd-path", type=str, default=None,
        help="USD scene file path (overrides TOML).",
    )
    run_all_parser.add_argument(
        "--input-video-dir", type=Path, default=None,
        help="Directory containing input videos for noise stage (overrides TOML).",
    )
    run_all_parser.add_argument(
        "--output-video-dir", type=Path, default=None,
        help="Directory to write noise-processed videos (overrides TOML).",
    )

    return parser


def _cmd_collect(args: argparse.Namespace) -> None:
    """Execute the collect subcommand."""
    from collect import DataCollector

    collector = DataCollector(
        num_samples=args.num_samples,
        video_duration_sec=args.duration,
        data_root=args.data_root,
        usd_path=args.usd_path,
    )
    collector.run()


def _cmd_noise(args: argparse.Namespace) -> None:
    """Execute the noise subcommand."""
    from data_modules.noise_processor import NoiseProcessor

    processor = NoiseProcessor(
        input_video_dir=args.input_video_dir,
        output_video_dir=args.output_video_dir,
    )
    processor.process_dataset()


def _cmd_json(args: argparse.Namespace) -> None:
    """Execute the json subcommand."""
    from data_modules.json_creation import JsonCreator

    creator = JsonCreator(data_root=args.data_root)
    creator.process_dataset()


def _cmd_run_all(args: argparse.Namespace) -> None:
    """Execute all pipeline stages sequentially."""
    from collect import DataCollector
    from data_modules.noise_processor import NoiseProcessor
    from data_modules.json_creation import JsonCreator

    print("=" * 60)
    print("STAGE 1: Data Collection")
    print("=" * 60)
    collector = DataCollector(
        num_samples=args.num_samples,
        video_duration_sec=args.duration,
        data_root=args.data_root,
        usd_path=args.usd_path,
    )
    collector.run()

    print("\n" + "=" * 60)
    print("STAGE 2: Noise Processing")
    print("=" * 60)
    processor = NoiseProcessor(
        input_video_dir=args.input_video_dir,
        output_video_dir=args.output_video_dir,
    )
    processor.process_dataset()

    print("\n" + "=" * 60)
    print("STAGE 3: JSON Metadata Generation")
    print("=" * 60)
    creator = JsonCreator(data_root=args.data_root)
    creator.process_dataset()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


_COMMANDS = {
    "collect": _cmd_collect,
    "noise": _cmd_noise,
    "json": _cmd_json,
    "run-all": _cmd_run_all,
}


def main() -> None:
    """Parse arguments and dispatch to the appropriate command handler."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handler = _COMMANDS[args.command]
    handler(args)


if __name__ == "__main__":
    main()
