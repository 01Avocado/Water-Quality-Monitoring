"""
Thin wrapper over the ThingSpeak REST API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests
from requests import Response

from .config import ThingSpeakConfig

logger = logging.getLogger(__name__)


class ThingSpeakError(RuntimeError):
    """Raised when ThingSpeak returns an error response."""


class ThingSpeakClient:
    """Utility class for querying ThingSpeak feeds."""

    def __init__(self, config: ThingSpeakConfig, session: Optional[requests.Session] = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        params = params or {}
        if self.config.read_api_key:
            params.setdefault("api_key", self.config.read_api_key)

        logger.debug("Requesting ThingSpeak endpoint %s with params=%s", url, params)
        response: Response = self.session.get(url, params=params, timeout=15)
        if response.status_code != 200:
            raise ThingSpeakError(
                f"ThingSpeak request failed: status={response.status_code}, body={response.text}"
            )

        payload = response.json()
        if isinstance(payload, dict) and payload.get("feeds") is None and payload.get("channel") is None:
            raise ThingSpeakError(f"Unexpected ThingSpeak payload: {payload}")
        return payload

    def fetch_latest(self, results: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch the latest feed entries for the configured channel.

        Returns the raw JSON payload produced by ThingSpeak, e.g.
        {
            "channel": {...},
            "feeds": [{...}, ...]
        }
        """
        limit = results if results is not None else self.config.result_limit
        endpoint = f"channels/{self.config.channel_id}/feeds.json"
        payload = self._request(endpoint, params={"results": limit})
        feeds: List[Dict[str, Any]] = payload.get("feeds", [])
        logger.debug("Received %d feed entries from ThingSpeak", len(feeds))
        return payload


