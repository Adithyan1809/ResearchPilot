"""ResearchLens - RAG-powered research paper analysis platform."""

import sys
from pathlib import Path

# Ensure both the repository root and the backend directory are in sys.path
# so that both `import app...` and `import backend.app...` resolve reliably.
_backend_dir = str(Path(__file__).resolve().parent.parent)
_root_dir = str(Path(__file__).resolve().parent.parent.parent)

for _path in [_backend_dir, _root_dir]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

__version__ = "1.0.0"
