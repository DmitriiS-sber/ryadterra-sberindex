"""Shared configuration loading for repeatable forecasting experiments.

The frozen YAML records the published experiment. Pass a copied YAML with
``--config`` and a separate output directory when testing other parameters.
Descriptive protocol fields are documented in docs/CONFIGURATION.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiment.yaml"
SUPPORTED_MODELS = {
    "last_value",
    "seasonal_naive",
    "damped_trend",
    "prophet_auto",
    "prophet_monthly",
    "pooled_ridge",
    "pooled_ridge_news",
    "chronos_bolt_tiny",
}


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Read YAML and reject parameters that cannot produce a valid forecast."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    models = config["models"]
    if not models or len(models) != len(set(models)):
        raise ValueError("models must contain unique model names")
    if set(models) - SUPPORTED_MODELS:
        raise ValueError(f"Unsupported models: {set(models) - SUPPORTED_MODELS}")
    if not config["origins"] or not config["horizons"]:
        raise ValueError("origins and horizons must be nonempty")
    if any(not isinstance(h, int) or not 1 <= h <= 12 for h in config["horizons"]):
        raise ValueError("Forecast horizons must be integers from 1 to 12")
    if config["workers"] < 1:
        raise ValueError("workers must be positive")
    trend = config["damped_trend"]
    if not 0 < trend["damping"] < 1 or trend["recent_months"] < 2:
        raise ValueError("damping must lie in (0, 1); recent_months must be >= 2")
    if config["ridge"]["minimum_lags"] < 6 or config["ridge"]["alpha"] < 0:
        raise ValueError("Ridge requires at least six lags and nonnegative alpha")
    if config["chronos"]["batch_size"] < 1:
        raise ValueError("Chronos batch_size must be positive")
    detectors = config["detectors"]
    if not 0 < detectors["alarm_false_positive_target"] < 1:
        raise ValueError("Detector false-positive target must lie in (0, 1)")
    if not 0 < detectors["ewma_alpha"] <= 1:
        raise ValueError("ewma_alpha must lie in (0, 1]")
    if detectors.get("reference_count", 12) < 1:
        raise ValueError("PH reference_count must be positive")
    return config


def foundation_revision(model_id: str) -> str | None:
    """Reuse the recorded snapshot revision only for its matching model ID."""
    path = ROOT / "configs/foundation_model.json"
    if not path.exists():
        return None
    pin = json.loads(path.read_text(encoding="utf-8"))
    return pin.get("revision") if pin.get("model_id") == model_id else None
