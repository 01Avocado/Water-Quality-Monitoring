"""
Entry point for running the realtime polling loop as a long-lived process.
"""

from __future__ import annotations

import logging

from .pipeline import RealtimePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def main() -> None:
    pipeline = RealtimePipeline()
    pipeline.run_forever()


if __name__ == "__main__":
    main()

