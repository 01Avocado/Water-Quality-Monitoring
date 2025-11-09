"""
Convert ThingSpeak feed entries into the feature set required by the models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .config import PipelineConfig


@dataclass
class NormalizedReading:
    timestamp: datetime
    pH: float
    turbidity: float
    temperature: float
    tds: float
    conductivity: Optional[float]
    dissolved_oxygen: Optional[float]
    raw_payload: Dict[str, Any]

    def as_dataframe(self) -> pd.DataFrame:
        data = {
            "timestamp": [self.timestamp],
            "pH": [self.pH],
            "Turbidity": [self.turbidity],
            "Temperature": [self.temperature],
            "TDS": [self.tds],
            "DO": [self.dissolved_oxygen],
        }
        return pd.DataFrame(data)


class SensorNormalizer:
    """Translate ThingSpeak fields into the feature set expected by the models."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.field_map = {k.lower(): v for k, v in config.thingspeak.field_map.items()}

    def _get(self, feed: Dict[str, Any], logical_name: str) -> Optional[float]:
        field_key = self.field_map.get(logical_name)
        if not field_key:
            return None
        raw_value = feed.get(field_key)
        if raw_value in (None, "", "null"):
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None

    def normalize(self, feed: Dict[str, Any]) -> Optional[NormalizedReading]:
        """Convert a ThingSpeak feed entry into NormalizedReading."""
        timestamp_raw = feed.get("created_at")
        try:
            timestamp = datetime.fromisoformat(timestamp_raw.rstrip("Z"))
        except Exception:
            timestamp = datetime.utcnow()

        temperature = self._get(feed, "temperature")
        pH = self._get(feed, "ph")
        conductivity = self._get(feed, "conductivity")
        turbidity = self._get(feed, "turbidity")
        dissolved_oxygen = self._get(feed, "dissolved_oxygen")

        if temperature is None or pH is None or turbidity is None:
            # Insufficient information to run downstream models
            return None

        tds = self._estimate_tds(conductivity)

        return NormalizedReading(
            timestamp=timestamp,
            pH=pH,
            turbidity=max(turbidity, 0.0),
            temperature=temperature,
            tds=tds,
            conductivity=conductivity,
            dissolved_oxygen=dissolved_oxygen,
            raw_payload=feed,
        )

    def _estimate_tds(self, conductivity: Optional[float]) -> float:
        if conductivity is None:
            # Fall back to a conservative default range if conductivity missing
            return 450.0
        return max(conductivity * self.config.conductivity_to_tds_factor, 0.0)


