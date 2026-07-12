"""Configuration loader for the portfolio optimization engine.

Reads `configs/portfolio.yaml` and returns a plain dict. Centralizes path
resolution so modules don't hard-code repo-relative locations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "portfolio.yaml"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load YAML config and return as dict.

    Parameters
    ----------
    path : Path-like, optional
        Override path; defaults to repo's `configs/portfolio.yaml`.

    Returns
    -------
    dict
        Parsed configuration.
    """
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def repo_path(*parts: str) -> Path:
    """Build an absolute path inside the repository."""
    return REPO_ROOT.joinpath(*parts)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Loaded config from {CONFIG_PATH}")
    print(f"Universe size: {len(cfg['universe']['vn30']) + len(cfg['universe']['diversification'])}")