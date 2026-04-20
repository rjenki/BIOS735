from pathlib import Path
from .config import OUTPUT_DIR, FIGURES_DIR, TABLES_DIR


def ensure_directories() -> None:
    for path in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in value).strip('_')
