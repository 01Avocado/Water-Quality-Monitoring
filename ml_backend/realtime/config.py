"""
Configuration utilities for the realtime ingestion and inference pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


DEFAULT_FIELD_MAP: Dict[str, str] = {
    # ThingSpeak field aliases -> logical sensor names used by the models
    "temperature": "field1",
    "ph": "field2",
    "conductivity": "field3",
    "turbidity": "field4",
    "dissolved_oxygen": "field5",
}


def _load_field_map() -> Dict[str, str]:
    """Load field mapping from environment if provided."""
    mapping_raw = os.getenv("THINGSPEAK_FIELD_MAP")
    if not mapping_raw:
        return DEFAULT_FIELD_MAP

    try:
        parsed = json.loads(mapping_raw)
        if not isinstance(parsed, dict):
            raise ValueError("Field map JSON must decode to a dictionary.")
        return {k.lower(): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError as exc:
        raise ValueError(
            "THINGSPEAK_FIELD_MAP must contain valid JSON, e.g. "
            '\'{"temperature": "field1", "ph": "field2"}\''
        ) from exc


@dataclass
class ThingSpeakConfig:
    """Configuration for connecting to ThingSpeak."""

    channel_id: str = field(default_factory=lambda: os.getenv("THINGSPEAK_CHANNEL_ID", ""))
    read_api_key: Optional[str] = field(default_factory=lambda: os.getenv("THINGSPEAK_READ_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("THINGSPEAK_BASE_URL", "https://api.thingspeak.com"))
    field_map: Dict[str, str] = field(default_factory=_load_field_map)
    result_limit: int = field(default_factory=lambda: int(os.getenv("THINGSPEAK_RESULT_LIMIT", "1")))
    poll_interval_seconds: int = field(default_factory=lambda: int(os.getenv("POLL_INTERVAL_SECONDS", "60")))

    def validate(self) -> None:
        if not self.channel_id:
            raise ValueError("ThingSpeak channel ID is required (set THINGSPEAK_CHANNEL_ID).")
        if not self.field_map:
            raise ValueError("Field map cannot be empty.")


@dataclass
class PipelineConfig:
    """High-level realtime pipeline configuration."""

    thingspeak: ThingSpeakConfig = field(default_factory=ThingSpeakConfig)
    conductivity_to_tds_factor: float = field(
        default_factory=lambda: float(os.getenv("CONDUCTIVITY_TO_TDS_FACTOR", "640.0"))
    )
    output_path: str = field(default_factory=lambda: os.getenv("REALTIME_OUTPUT_PATH", "ml_backend/realtime/latest_output.json"))
    unsafe_wqi_threshold: float = field(
        default_factory=lambda: float(os.getenv("UNSAFE_WQI_THRESHOLD", "50.0"))
    )
    dashboard_webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("DASHBOARD_WEBHOOK_URL"))

    def validate(self) -> None:
        self.thingspeak.validate()
        if self.conductivity_to_tds_factor <= 0:
            raise ValueError("CONDUCTIVITY_TO_TDS_FACTOR must be positive.")


def load_pipeline_config() -> PipelineConfig:
    """Convenience helper to load and validate configuration."""
    config = PipelineConfig()
    config.validate()
    return config


