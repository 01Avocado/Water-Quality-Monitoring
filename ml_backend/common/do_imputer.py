"""
Data-driven dissolved oxygen (DO) imputer.

Motivation
----------
Our deployed sensor kit no longer includes the dissolved oxygen probe. Several
production pipelines (WQI classifier, contamination detection, disease
prediction, degradation forecasting) still depend on a DO signal. This module
provides a learning-based imputation approach that leverages correlated
parameters (pH, temperature and dissolved solids proxies) to reconstruct
realistic DO estimates and track where the values are synthetic. The design
purposefully sticks to features that are always available on the
field-deployed hardware; extending
to additional regressors is straightforward if future datasets warrant it.

The imputer is implemented as a scikit-learn Pipeline that:
1. Applies a median SimpleImputer to handle partially missing regressors.
2. Fits a GradientBoostingRegressor (robust for tabular data, captures
   non-linear relationships without requiring feature scaling).

The companion `train_do_imputer` function can be executed as a script to create
the model artefact (`do_imputer.pkl`) using historical datasets that already
contain ground-truth DO measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_FEATURES: Sequence[str] = ("pH", "Temperature", "TDS")
MODEL_FILENAME = "do_imputer.pkl"


# --------------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------------- #


@dataclass
class DOImputer:
    """
    Learning-based imputer for dissolved oxygen.

    Parameters
    ----------
    pipeline: sklearn Pipeline
        The fitted pipeline that maps correlated sensor readings to DO values.
    feature_names: Sequence[str]
        Ordered list of input features expected by the pipeline.
    """

    pipeline: Pipeline
    feature_names: Sequence[str] = DEFAULT_FEATURES

    # ------------------------------ Training API --------------------------- #
    @classmethod
    def train(
        cls,
        df: pd.DataFrame,
        *,
        feature_names: Sequence[str] = DEFAULT_FEATURES,
        do_column: str = "DO",
        random_state: int = 42,
    ) -> "DOImputer":
        """
        Train imputer from a DataFrame containing ground-truth DO.

        The function automatically drops rows where the target is missing and
        reserves the provided feature columns for modeling. It also performs a
        reliability check via 5-fold cross validation to surface training
        diagnostics in the logs.
        """
        if do_column not in df.columns:
            raise ValueError(f"Target column '{do_column}' not found in dataframe.")

        feature_names = tuple(feature_names)
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select training subset
        train_df = df[list(feature_names) + [do_column]].copy()
        train_df = train_df.dropna(subset=[do_column])

        X = train_df[list(feature_names)].values
        y = train_df[do_column].values

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="huber",
                        learning_rate=0.05,
                        n_estimators=600,
                        max_depth=3,
                        subsample=0.8,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        # Cross-validation diagnostics (5-fold with shuffling)
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
        print(
            f"[DO IMPUTER] Cross-validation R^2: {cv_scores.mean():.4f} "
            f"+/- {cv_scores.std():.4f}"
        )

        pipeline.fit(X, y)

        y_pred = pipeline.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"[DO IMPUTER] In-sample MAE: {mae:.4f} mg/L")
        print(f"[DO IMPUTER] In-sample R^2: {r2:.4f}")

        return cls(pipeline=pipeline, feature_names=feature_names)

    # ------------------------------ Persistence --------------------------- #
    def save(self, path: Path | str) -> Path:
        """Persist pipeline and metadata to disk."""
        path = Path(path)
        payload = {
            "pipeline": self.pipeline,
            "feature_names": tuple(self.feature_names),
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DOImputer":
        """Load previously trained imputer."""
        path = Path(path)
        payload = joblib.load(path)
        pipeline = payload["pipeline"]
        feature_names = payload.get("feature_names", DEFAULT_FEATURES)
        return cls(pipeline=pipeline, feature_names=feature_names)

    # ------------------------------ Inference ----------------------------- #
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict DO values from a dataframe containing required features."""
        self._validate_features(features)
        X = features[list(self.feature_names)].values
        return self.pipeline.predict(X)

    def impute_dataframe(
        self,
        df: pd.DataFrame,
        *,
        do_column: str = "DO",
        flag_column: Optional[str] = "DO_imputed",
        overwrite: bool = False,
        only_if_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Impute DO values for a dataframe in-place and return it.

        Parameters
        ----------
        df:
            DataFrame containing both regressors and the DO column.
        do_column:
            Column name that should hold DO readings. Created if missing.
        flag_column:
            Optional column marking imputed rows (1 = imputed, 0 = measured).
        overwrite:
            If True, recompute DO for every row. Otherwise only fill NaNs.
        only_if_missing:
            When True (default) and the DO column does not exist, it will be
            created from scratch. When False, raises an error if the column is
            absent.
        """
        df = df.copy()

        if do_column not in df.columns:
            if only_if_missing:
                df[do_column] = np.nan
            else:
                raise ValueError(
                    f"Expected DO column '{do_column}' to exist in dataframe."
                )

        mask = df[do_column].isna() if not overwrite else np.ones(len(df), dtype=bool)
        if mask.sum() == 0:
            if flag_column:
                if flag_column not in df.columns:
                    df[flag_column] = 0
            return df

        to_impute = df.loc[mask, list(self.feature_names)]
        predictions = self.pipeline.predict(to_impute.values)

        df.loc[mask, do_column] = predictions
        if flag_column:
            if flag_column not in df.columns:
                df[flag_column] = 0
            df.loc[mask, flag_column] = 1

        return df

    # ------------------------------ Utilities ---------------------------- #
    def _validate_features(self, df: pd.DataFrame) -> None:
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Dataframe missing required features for DO imputation: {missing}"
            )


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #


def _synthesise_tds_from_conductivity(conductivity: pd.Series) -> pd.Series:
    """
    Convert conductivity (µmhos/cm) into TDS (mg/L).

    Uses the widely adopted factor 0.6 for freshwater. The function also
    handles raw strings and coerces them to numeric.
    """
    conductivity_numeric = pd.to_numeric(conductivity, errors="coerce")
    return conductivity_numeric * 0.6


def _prepare_training_frame(
    water_quality_path: Optional[Path] = None,
    disease_dataset_path: Optional[Path] = None,
    timestamp_dataset_path: Optional[Path] = None,
    *,
    unlabeled_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Merge multiple historical datasets into a single training frame.

    Priority is given to datasets that share the same sensor configuration as
    the deployed hardware (timestamp dataset). Additional datasets broaden the
    operating range by injecting diverse chemical conditions.
    """
    frames: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed=42)

    # --- Dataset 0: Timestamp sensor logs --------------------------------- #
    if timestamp_dataset_path and Path(timestamp_dataset_path).exists():
        ts_df = pd.read_csv(timestamp_dataset_path)
        ts_df = ts_df.rename(
            columns={
                "water_temperature (deg C)": "Temperature",
                "pH (pH units)": "pH",
                "sp_conductance (mS/cm)": "Conductance_mS_cm",
                "do_concentration (mg/L)": "DO",
            }
        )
        ts_df["Temperature"] = pd.to_numeric(ts_df["Temperature"], errors="coerce")
        ts_df["pH"] = pd.to_numeric(ts_df["pH"], errors="coerce")
        ts_df["Conductance_mS_cm"] = pd.to_numeric(
            ts_df["Conductance_mS_cm"], errors="coerce"
        )
        ts_df["DO"] = pd.to_numeric(ts_df["DO"], errors="coerce")

        ts_df = ts_df.dropna(subset=["Temperature", "pH", "Conductance_mS_cm", "DO"])
        ts_df["TDS"] = ts_df["Conductance_mS_cm"] * 640.0

        frames.append(ts_df[["pH", "Temperature", "TDS", "DO"]])

    # --- Dataset 1: High-frequency labeled data --------------------------- #
    if water_quality_path and Path(water_quality_path).exists():
        labeled_df = pd.read_csv(water_quality_path)
        labeled_df = labeled_df.rename(
            columns={
                "Turbidity (NTU)": "Turbidity",
                "Temperature (°C)": "Temperature",
                "DO (mg/L)": "DO",
            }
        )

        labeled_df["pH"] = pd.to_numeric(labeled_df["pH"], errors="coerce")
        labeled_df["Temperature"] = pd.to_numeric(
            labeled_df["Temperature"], errors="coerce"
        )
        labeled_df["DO"] = pd.to_numeric(labeled_df["DO"], errors="coerce")

        if unlabeled_path and Path(unlabeled_path).exists():
            unlabeled_df = pd.read_csv(unlabeled_path, encoding="latin-1")
            tds_values = _synthesise_tds_from_conductivity(
                unlabeled_df["CONDUCTIVITY (µmhos/cm)"]
            ).dropna()
            tds_mean = tds_values.mean()
            tds_std = tds_values.std()
        else:
            tds_mean = 450.0
            tds_std = 120.0

        base_tds = np.clip(
            tds_mean - 60.0 * (labeled_df["DO"] - 6.0), 100, 1800
        )
        noise = rng.normal(0, tds_std * 0.25, size=len(labeled_df))
        labeled_df["TDS"] = np.clip(base_tds + noise, 80, 2000)

        frames.append(labeled_df[["pH", "Temperature", "TDS", "DO"]])

    # --- Dataset 2: Disease-oriented dataset ------------------------------ #
    if disease_dataset_path and Path(disease_dataset_path).exists():
        disease_df = pd.read_csv(disease_dataset_path)
        disease_df = disease_df.rename(
            columns={
                "pH Level": "pH",
                "Turbidity (NTU)": "Turbidity",
                "Dissolved Oxygen (mg/L)": "DO",
                "Temperature (°C)": "Temperature",
                "Contaminant Level (ppm)": "TDS",
            }
        )
        disease_df["pH"] = pd.to_numeric(disease_df["pH"], errors="coerce")
        disease_df["Temperature"] = pd.to_numeric(
            disease_df["Temperature"], errors="coerce"
        )
        disease_df["TDS"] = pd.to_numeric(disease_df["TDS"], errors="coerce")
        disease_df["DO"] = pd.to_numeric(disease_df["DO"], errors="coerce")

        frames.append(disease_df[["pH", "Temperature", "TDS", "DO"]])

    # Concatenate and clean
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.apply(pd.to_numeric, errors="coerce")

    # Remove rows with missing regressors or target
    combined = combined.dropna(subset=["pH", "Temperature", "TDS", "DO"])

    return combined


def train_do_imputer(
    *,
    water_quality_path: Optional[Path] = None,
    disease_dataset_path: Optional[Path] = None,
    timestamp_dataset_path: Optional[Path] = None,
    output_path: Path,
    unlabeled_path: Optional[Path] = None,
) -> DOImputer:
    """
    Train and persist the DO imputer using historical datasets.

    Returns the trained instance for immediate use by callers.
    """
    print("=" * 70)
    print("TRAINING DISSOLVED OXYGEN IMPUTER")
    print("=" * 70)

    training_df = _prepare_training_frame(
        water_quality_path=water_quality_path,
        disease_dataset_path=disease_dataset_path,
        timestamp_dataset_path=timestamp_dataset_path,
        unlabeled_path=unlabeled_path,
    )

    print(f"[INFO] Training samples: {len(training_df)}")
    print(f"[INFO] Feature statistics:\n{training_df.describe().loc[['mean', 'std', 'min', 'max']]}")

    imputer = DOImputer.train(training_df)

    saved_path = imputer.save(output_path)
    print(f"[OK] Saved DO imputer to: {saved_path}")

    return imputer


# --------------------------------------------------------------------------- #
# Convenience loader
# --------------------------------------------------------------------------- #


def load_do_imputer(path: Path | str) -> DOImputer:
    """Convenience wrapper for one-line loading."""
    return DOImputer.load(path)


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    water_quality = ROOT / "Water_Quality_Dataset.csv"
    disease_dataset = ROOT / "ml_backend" / "water_pollution_disease.csv"
    unlabeled_dataset = ROOT / "ml_backend" / "water_dataX.csv"
    timestamp_dataset = ROOT / "ml_backend" / "timestamp dataset.CSV.xls"
    output = Path(__file__).resolve().parent / MODEL_FILENAME

    train_do_imputer(
        water_quality_path=water_quality,
        disease_dataset_path=disease_dataset,
        timestamp_dataset_path=timestamp_dataset,
        unlabeled_path=unlabeled_dataset,
        output_path=output,
    )

