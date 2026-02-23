"""Colab training entrypoint using NumerAI official NumerAPI dataset flow."""

from __future__ import annotations

import logging

from config import _optional_bool_env


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    if _optional_bool_env("TRAIN_DRY_RUN", default=False):
        from training_dry_run import train_dry_run

        train_dry_run()
    else:
        from training_pipeline import train

        train()
