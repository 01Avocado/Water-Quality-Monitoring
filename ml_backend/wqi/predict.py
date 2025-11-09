"""
Inference utilities for the WQI classifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np


@dataclass
class WQIPredictorConfig:
    model_path: str = "ml_backend/wqi/wqi_model.pkl"
    scaler_path: str = "ml_backend/wqi/wqi_scaler.pkl"
    config_path: str = "ml_backend/wqi/wqi_config.pkl"


class WQIPredictor:
    """Lightweight wrapper around the trained WQI model."""

    def __init__(self, config: WQIPredictorConfig | None = None) -> None:
        self.config = config or WQIPredictorConfig()
        self.model = joblib.load(self.config.model_path)
        self.scaler = joblib.load(self.config.scaler_path)
        self.model_config: Dict[str, Dict[str, float]] = joblib.load(self.config.config_path)

        self.feature_names = self.model_config["feature_names"]
        self.who_weights = self.model_config["WHO_WEIGHTS"]
        self.ideal_ranges = self.model_config["IDEAL_RANGES"]
        self.class_names = self.model_config["class_names"]

    def _compute_wqi_score(self, features: np.ndarray) -> np.ndarray:
        wqi_scores = np.zeros(len(features))

        for idx, feature_name in enumerate(self.feature_names):
            values = features[:, idx]
            weight = self.who_weights[feature_name]
            ideal_min, ideal_max = self.ideal_ranges[feature_name]

            sub_index = np.zeros(len(values))
            for j, value in enumerate(values):
                if ideal_min <= value <= ideal_max:
                    ideal_center = (ideal_min + ideal_max) / 2
                    ideal_range = ideal_max - ideal_min
                    deviation = abs(value - ideal_center) / (ideal_range / 2)
                    sub_index[j] = 100 * (1 - deviation * 0.2)
                else:
                    if value < ideal_min:
                        deviation = (ideal_min - value) / ideal_min if ideal_min != 0 else value
                    else:
                        deviation = (value - ideal_max) / ideal_max if ideal_max != 0 else value
                    sub_index[j] = 100 * np.exp(-deviation)

            wqi_scores += weight * sub_index

        return wqi_scores

    def predict(self, features: Dict[str, float]) -> Dict[str, float | int | str]:
        """
        Run the WQI classifier on a single reading.

        Args:
            features: dict containing pH, Turbidity, TDS, DO, Temperature.
        """
        feature_vector = np.array([[features["pH"], features["Turbidity"], features["TDS"], features["DO"], features["Temperature"]]])
        wqi_score = self._compute_wqi_score(feature_vector).reshape(-1, 1)
        enhanced = np.hstack([feature_vector, wqi_score])
        scaled = self.scaler.transform(enhanced)

        probs = self.model.predict_proba(scaled)[0]
        label_idx = int(np.argmax(probs))
        return {
            "wqi_score": float(wqi_score[0][0]),
            "pollution_level": int(self.model.predict(scaled)[0]),
            "status": self.class_names[label_idx],
            "probabilities": {self.class_names[i]: float(prob) for i, prob in enumerate(probs)},
        }


