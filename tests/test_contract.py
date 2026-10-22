"""Contract tests for Add/Search."""
from __future__ import annotations

HEADERS = {"Authorization": "Bearer test-key"}


def add_payload(
    request_id: str = "r1",
    user_id: str = "u1",
    session_id: str = "s1",
    messages: list[dict] | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages
        or [
            {
                "role": "user",
                "content": "Alice loves jasmine tea.",
                "timestamp": 1704067200000,
            },
            {"role": "assistant", "content": "Noted."},
        ],
    }


async def test_health(client) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_add_requires_auth(client) -> None:
    response = await client.post("/add", json=add_payload())
    assert response.status_code == 401


async def test_add_and_search(client) -> None:
    response = await client.post("/add", json=add_payload(), headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "request_id": "r1",
        "user_id": "u1",
        "session_id": "s1",
    }

    search = await client.post(
        "/search",
        json={"query": "jasmine tea", "user_id": "u1", "top_k": 10},
        headers=HEADERS,
    )
    assert search.status_code == 200
    data = search.json()["data"]
    assert data
    assert any("jasmine" in item["content"].lower() for item in data)
    for item in data:
        assert item["id"]
        assert item["content"]
        assert isinstance(item["score"], float)


async def test_idempotent_replay(client, service) -> None:
    payload = add_payload()
    first = await client.post("/add", json=payload, headers=HEADERS)
    second = await client.post("/add", json=payload, headers=HEADERS)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()

    async with service.db.pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM chunks WHERE user_id = 'u1'")
    assert count == 1


async def test_conflicting_payload(client) -> None:
    await client.post("/add", json=add_payload(), headers=HEADERS)
    changed = add_payload()
    changed["messages"][0]["content"] = "Alice loves coffee."
    response = await client.post("/add", json=changed, headers=HEADERS)
    assert response.status_code == 409


async def test_user_isolation(client) -> None:
    await client.post("/add", json=add_payload(), headers=HEADERS)
    await client.post(
        "/add",
        json=add_payload(
            request_id="r2",
            user_id="u2",
            session_id="s2",
            messages=[{"role": "user", "content": "Bob loves coffee."}],
        ),
        headers=HEADERS,
    )

    response = await client.post(
        "/search",
        json={"query": "coffee", "user_id": "u2", "top_k": 10},
        headers=HEADERS,
    )
    data = response.json()["data"]
    assert data
    assert all("Alice" not in item["content"] for item in data)


async def test_top_k_is_respected(client) -> None:
    messages = [
        {"role": "user", "content": f"fact number {index} about topic {index}"}
        for index in range(12)
    ]
    await client.post(
        "/add",
        json=add_payload(
            request_id="rk", user_id="uk", session_id="sk", messages=messages
        ),
        headers=HEADERS,
    )
    response = await client.post(
        "/search",
        json={"query": "fact number 3", "user_id": "uk", "top_k": 1},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


async def test_empty_search(client) -> None:
    response = await client.post(
        "/search",
        json={"query": "anything", "user_id": "nobody", "top_k": 5},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


async def test_missing_field_is_422(client) -> None:
    response = await client.post(
        "/add",
        json={"request_id": "r", "messages": [{"role": "user", "content": "x"}]},
        headers=HEADERS,
    )
    assert response.status_code == 422
