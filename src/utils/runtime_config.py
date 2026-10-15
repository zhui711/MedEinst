"""Small path and endpoint helpers for the public launchers."""

import os
from pathlib import Path


def configured_base_url(section: dict):
    """An optional environment override takes precedence over the YAML URL."""
    variable = section.get("base_url_env")
    return (os.getenv(variable) if variable else None) or section.get("base_url") or None

def config_relative_path(value: str, config_path: Path) -> str:
    """Resolve a YAML path relative to its configuration file."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())
