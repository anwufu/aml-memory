"""Local end-to-end smoke test against a running service.

Usage:
    python scripts/smoke_local.py [base_url]

The Memory System Key is read from .env (MEMORY_API_KEY).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_memory.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    headers = {"Authorization": f"Bearer {settings.memory_api_key}"}

    with httpx.Client(timeout=60) as client:
        health = client.get(f"{base_url}/health")
        print("health:", health.status_code, health.text)

        add_body = {
            "request_id": "smoke:local:1",
            "user_id": "smoke:user",
            "session_id": "smoke:session",
            "messages": [
                {
                    "role": "user",
                    "content": "My sister Ada lives in Hangzhou and loves jasmine tea.",
                    "timestamp": 1704067200000,
                },
                {"role": "assistant", "content": "I will remember that."},
                {
                    "role": "user",
                    "content": "She moved to Suzhou last month for a new job.",
                    "timestamp": 1704153600000,
                },
            ],
        }
        added = client.post(f"{base_url}/add", json=add_body, headers=headers)
        print("add:", added.status_code, added.text)

        replay = client.post(f"{base_url}/add", json=add_body, headers=headers)
        print("add replay:", replay.status_code, replay.text)

        search_body = {
            "query": "Where does Ada live now?",
            "user_id": "smoke:user",
            "top_k": 10,
        }
        found = client.post(f"{base_url}/search", json=search_body, headers=headers)
        print("search:", found.status_code)
        print(json.dumps(found.json(), ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
