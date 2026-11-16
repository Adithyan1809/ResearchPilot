"""Convenience script to run the ResearchLens development server.

Usage::

    cd backend
    python run.py

    # or from repository root:
    python backend/run.py
"""

import sys
from pathlib import Path

# Ensure paths are configured before imports
_backend_dir = Path(__file__).resolve().parent
_root_dir = _backend_dir.parent

for p in [str(_backend_dir), str(_root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import uvicorn
from backend.app.config import get_settings
from backend.app.main import app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
