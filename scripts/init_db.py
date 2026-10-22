"""Create the database schema."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_memory.config import get_settings  # noqa: E402
from aml_memory.db import Database  # noqa: E402


async def main() -> None:
    database = Database(get_settings())
    await database.connect()
    await database.init_schema()
    await database.close()
    print("schema ready")


if __name__ == "__main__":
    asyncio.run(main())
