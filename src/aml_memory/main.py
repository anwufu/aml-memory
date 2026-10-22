"""Entry point: python -m aml_memory.main"""
from __future__ import annotations

import uvicorn

from .config import get_settings
from .api import app


def main() -> None:
    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", workers=1)


if __name__ == "__main__":
    main()
