"""
Realtime pipeline that pulls data from ThingSpeak and runs all models.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import requests

from .config import PipelineConfig, load_pipeline_config
from .model_service import ModelService
from .normalizer import SensorNormalizer
from .thing_speak_client import ThingSpeakClient

logger = logging.getLogger(__name__)


class RealtimePipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_service: Optional[ModelService] = None,
        thingspeak_client: Optional[ThingSpeakClient] = None,
        normalizer: Optional[SensorNormalizer] = None,
    ) -> None:
        self.config = config or load_pipeline_config()
        self.normalizer = normalizer or SensorNormalizer(self.config)
        self.client = thingspeak_client or ThingSpeakClient(self.config.thingspeak)
        self.models = model_service or ModelService()
        self.last_entry_id: Optional[int] = None

    def run_once(self) -> Optional[Dict[str, object]]:
        payload = self.client.fetch_latest(results=1)
        feeds = payload.get("feeds") or []
        if not feeds:
            logger.warning("ThingSpeak returned no feeds.")
            return None

        latest = feeds[-1]
        entry_id = latest.get("entry_id")
        if entry_id is not None and entry_id == self.last_entry_id:
            logger.info("No new data since last poll (entry_id=%s).", entry_id)
            return None

        normalized = self.normalizer.normalize(latest)
        if normalized is None:
            logger.warning("Feed entry lacked required sensor values, skipping.")
            return None

        results = self.models.predict(normalized)
        results["raw_channel"] = payload.get("channel", {})
        results["raw_feed"] = latest
        self._persist(results)

        self.last_entry_id = entry_id
        return results

    def run_forever(self) -> None:
        logger.info("Starting realtime loop (interval=%ss)", self.config.thingspeak.poll_interval_seconds)
        while True:
            try:
                self.run_once()
            except Exception:  # pylint: disable=broad-except
                logger.exception("Realtime pipeline iteration failed.")
            time.sleep(self.config.thingspeak.poll_interval_seconds)

    def _persist(self, results: Dict[str, object]) -> None:
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Wrote realtime output to %s", output_path)

        if self.config.dashboard_webhook_url:
            try:
                response = requests.post(
                    self.config.dashboard_webhook_url,
                    json=results,
                    timeout=10,
                )
                response.raise_for_status()
                logger.info("Published results to dashboard webhook: %s", self.config.dashboard_webhook_url)
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to publish results to dashboard webhook.")


