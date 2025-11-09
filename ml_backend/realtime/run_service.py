"""
Command-line entrypoint for the realtime pipeline.
"""

from __future__ import annotations

import argparse
import logging

from .pipeline import RealtimePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run realtime ThingSpeak ingestion and inference.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously instead of processing a single batch.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = RealtimePipeline()
    if args.loop:
        pipeline.run_forever()
    else:
        result = pipeline.run_once()
        if result is None:
            logging.info("No new data processed.")
        else:
            logging.info("Realtime inference complete.")


if __name__ == "__main__":
    main()


