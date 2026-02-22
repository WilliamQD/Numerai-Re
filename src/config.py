"""Runtime configuration parsing for training and inference entrypoints."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainRuntimeConfig:
    dataset_version: str = "v5.2"
    feature_set_name: str = "medium"
    target_col: str = "target"
    era_col: str = "era"
    model_name: str = "lgbm_numerai_v52"
    num_boost_round: int = 5000
    early_stopping_rounds: int = 250
    wandb_project: str = "numerai-mlops"
    wandb_entity: str | None = None
    wandb_api_key: str = ""
    numerai_public_id: str | None = None
    numerai_secret_key: str | None = None
    numerai_data_dir: Path = Path("/content/numerai_data")
    lgbm_device: str = "gpu"

    @classmethod
    def from_env(cls) -> "TrainRuntimeConfig":
        wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()
        if not wandb_api_key:
            raise RuntimeError(
                "WANDB_API_KEY environment variable is not set. "
                "Please set it before running training to enable Weights & Biases logging."
            )

        lgbm_device = os.getenv("LGBM_DEVICE", "gpu").strip().lower() or "gpu"
        if lgbm_device not in {"gpu", "cpu"}:
            raise ValueError("Invalid LGBM_DEVICE. Expected one of: gpu, cpu.")
        numerai_public_id = os.getenv("NUMERAI_PUBLIC_ID", "").strip() or None
        numerai_secret_key = os.getenv("NUMERAI_SECRET_KEY", "").strip() or None
        if bool(numerai_public_id) != bool(numerai_secret_key):
            raise RuntimeError(
                "NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY must be set together for authenticated training downloads."
            )
        dataset_version = os.getenv("NUMERAI_DATASET_VERSION", "v5.2").strip() or "v5.2"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_version):
            raise ValueError("Invalid NUMERAI_DATASET_VERSION. Use only letters, numbers, dot, underscore, or hyphen.")
        if ".." in dataset_version or dataset_version.startswith((".", "-")):
            raise ValueError("Invalid NUMERAI_DATASET_VERSION.")

        return cls(
            dataset_version=dataset_version,
            feature_set_name=os.getenv("NUMERAI_FEATURE_SET", "medium"),
            model_name=os.getenv("WANDB_MODEL_NAME", "lgbm_numerai_v52"),
            wandb_project=os.getenv("WANDB_PROJECT", "numerai-mlops"),
            wandb_entity=os.getenv("WANDB_ENTITY"),
            wandb_api_key=wandb_api_key,
            numerai_public_id=numerai_public_id,
            numerai_secret_key=numerai_secret_key,
            numerai_data_dir=Path(os.getenv("NUMERAI_DATA_DIR", "/content/numerai_data")),
            lgbm_device=lgbm_device,
        )


@dataclass(frozen=True)
class InferenceRuntimeConfig:
    numerai_public_id: str
    numerai_secret_key: str
    numerai_model_name: str
    wandb_entity: str
    wandb_project: str
    dataset_version: str = "v5.2"
    wandb_model_name: str = "lgbm_numerai_v52"
    min_pred_std: float = 1e-6
    max_abs_exposure: float = 0.30

    @classmethod
    def from_env(cls) -> "InferenceRuntimeConfig":
        min_pred_std_str = os.getenv("MIN_PRED_STD", "1e-6")
        try:
            min_pred_std = float(min_pred_std_str)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid MIN_PRED_STD value {min_pred_std_str!r}: must be a valid float"
            ) from exc

        max_abs_exposure_str = os.getenv("MAX_ABS_EXPOSURE", "0.30")
        try:
            max_abs_exposure = float(max_abs_exposure_str)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid MAX_ABS_EXPOSURE value {max_abs_exposure_str!r}: must be a valid float"
            ) from exc

        return cls(
            numerai_public_id=_required_env("NUMERAI_PUBLIC_ID"),
            numerai_secret_key=_required_env("NUMERAI_SECRET_KEY"),
            numerai_model_name=_required_env("NUMERAI_MODEL_NAME"),
            wandb_entity=_required_env("WANDB_ENTITY"),
            wandb_project=_required_env("WANDB_PROJECT"),
            dataset_version=os.getenv("NUMERAI_DATASET_VERSION", "v5.2"),
            wandb_model_name=os.getenv("WANDB_MODEL_NAME", "lgbm_numerai_v52"),
            min_pred_std=min_pred_std,
            max_abs_exposure=max_abs_exposure,
        )


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
