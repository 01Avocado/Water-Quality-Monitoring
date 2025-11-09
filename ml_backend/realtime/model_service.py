"""
Aggregate all pre-trained models for realtime inference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import pandas as pd

from ..common import load_do_imputer
from ..common.do_imputer import DOImputer
from ..contamination_detection.predict import ContaminationPredictor
from ..disease_outbreak.predict import DiseaseOutbreakPredictor
from ..wqi.predict import WQIPredictor, WQIPredictorConfig
from .normalizer import NormalizedReading


@dataclass
class ModelPaths:
    wqi_model: str = "ml_backend/wqi/wqi_model.pkl"
    wqi_scaler: str = "ml_backend/wqi/wqi_scaler.pkl"
    wqi_config: str = "ml_backend/wqi/wqi_config.pkl"

    contamination_model: str = "ml_backend/contamination_detection/contamination_model.pkl"
    contamination_imputer: str = "ml_backend/contamination_detection/imputer.pkl"
    contamination_encoder: str = "ml_backend/contamination_detection/label_encoder.pkl"

    disease_model: str = "ml_backend/disease_outbreak/disease_model.pkl"
    disease_scaler: str = "ml_backend/disease_outbreak/disease_scaler.pkl"
    disease_config: str = "ml_backend/disease_outbreak/disease_config.pkl"

    degradation_model: str = "ml_backend/degradation_forecasting/degradation_model.h5"
    degradation_scaler: str = "ml_backend/degradation_forecasting/degradation_scaler.pkl"
    degradation_config: str = "ml_backend/degradation_forecasting/degradation_config.pkl"

    do_imputer: str = "ml_backend/common/do_imputer.pkl"


@dataclass
class ModelService:
    """Load once, predict often."""

    paths: ModelPaths = field(default_factory=ModelPaths)
    wqi: WQIPredictor = field(init=False)
    contamination: ContaminationPredictor = field(init=False)
    disease: DiseaseOutbreakPredictor = field(init=False)
    degradation: Optional[object] = field(init=False)
    do_imputer: DOImputer = field(init=False)
    history: Deque[Dict[str, float]] = field(init=False)
    degradation_lookback: int = field(init=False)
    degradation_available: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.do_imputer = load_do_imputer(self.paths.do_imputer)
        self.wqi = WQIPredictor(
            WQIPredictorConfig(
                model_path=self.paths.wqi_model,
                scaler_path=self.paths.wqi_scaler,
                config_path=self.paths.wqi_config,
            )
        )
        self.contamination = ContaminationPredictor(
            model_path=self.paths.contamination_model,
            imputer_path=self.paths.contamination_imputer,
            encoder_path=self.paths.contamination_encoder,
            do_imputer_path=self.paths.do_imputer,
        )
        self.disease = DiseaseOutbreakPredictor(
            model_path=self.paths.disease_model,
            scaler_path=self.paths.disease_scaler,
            config_path=self.paths.disease_config,
            do_imputer_path=self.paths.do_imputer,
        )
        try:
            from ..degradation_forecasting.predict import DegradationPredictor  # type: ignore

            self.degradation = DegradationPredictor(
                model_path=self.paths.degradation_model,
                scaler_path=self.paths.degradation_scaler,
                config_path=self.paths.degradation_config,
                do_imputer_path=self.paths.do_imputer,
            )
            self.degradation_lookback = self.degradation.lookback  # type: ignore[attr-defined]
            self.history = deque(maxlen=self.degradation_lookback)
        except ModuleNotFoundError as exc:
            # TensorFlow is optional in some environments; degrade gracefully.
            self.degradation = None
            self.degradation_available = False
            self.degradation_error = str(exc)
            self.degradation_lookback = 0
            self.history = deque()

    def _impute_do(self, reading: NormalizedReading) -> Dict[str, float]:
        df = reading.as_dataframe()
        df = self.do_imputer.impute_dataframe(
            df,
            do_column="DO",
            flag_column="DO_imputed",
            only_if_missing=True,
        )
        row = df.iloc[0]
        return {
            "pH": float(row["pH"]),
            "Turbidity": float(row["Turbidity"]),
            "Temperature": float(row["Temperature"]),
            "TDS": float(row["TDS"]),
            "DO": float(row["DO"]),
            "DO_imputed_flag": int(row.get("DO_imputed", 0)),
        }

    def predict(self, reading: NormalizedReading) -> Dict[str, object]:
        enriched = self._impute_do(reading)

        # WQI
        wqi_result = self.wqi.predict(enriched)

        # Contamination
        contamination_result = self.contamination.predict_single(
            pH=enriched["pH"],
            turbidity=enriched["Turbidity"],
            tds=enriched["TDS"],
            do=enriched["DO"],
            temperature=enriched["Temperature"],
        )

        # Disease outbreak
        disease_result = self.disease.predict_single(
            pH=enriched["pH"],
            turbidity=enriched["Turbidity"],
            tds=enriched["TDS"],
            do=enriched["DO"],
            temperature=enriched["Temperature"],
        )

        # Degradation forecasting (requires rolling history)
        history_entry = {
            "pH": enriched["pH"],
            "Turbidity": enriched["Turbidity"],
            "TDS": enriched["TDS"],
            "DO": enriched["DO"],
            "Temperature": enriched["Temperature"],
        }
        if self.degradation_available:
            self.history.append(history_entry)

        degradation_report: Optional[Dict[str, object]] = None
        if self.degradation_available and len(self.history) == self.degradation_lookback:
            history_df = pd.DataFrame(list(self.history))
            degradation_report = self.degradation.get_full_forecast_report(  # type: ignore[union-attr]
                history_df[self.degradation.feature_names],
                current_wqi=wqi_result["wqi_score"],
            )
        elif not self.degradation_available:
            degradation_report = {
                "status": "unavailable",
                "message": "Degradation forecasting disabled (TensorFlow not installed).",
            }
        elif len(self.history) < self.degradation_lookback:
            degradation_report = {
                "status": "insufficient_history",
                "message": f"Need {self.degradation_lookback} readings for degradation forecast.",
                "required_readings": self.degradation_lookback - len(self.history),
            }

        return {
            "timestamp": reading.timestamp.isoformat(),
            "sensor_values": {
                "pH": enriched["pH"],
                "Turbidity": enriched["Turbidity"],
                "Temperature": enriched["Temperature"],
                "TDS": enriched["TDS"],
                "DO": enriched["DO"],
                "DO_imputed": bool(enriched["DO_imputed_flag"]),
            },
            "wqi": wqi_result,
            "contamination": contamination_result,
            "disease": disease_result,
            "degradation": degradation_report,
        }


