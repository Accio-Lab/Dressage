from __future__ import annotations

import asyncio

import httpx

from dressage.proxy.proxy_client import ProxyClient


def test_proxy_client_sends_default_headers_and_reads_capabilities():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/integration/capabilities":
            return httpx.Response(
                200,
                json={
                    "schema_version": "dressage.proxy.integration/v1",
                    "current_weight_version": "weights-v7",
                },
            )
        return httpx.Response(200, json={"ok": True})

    async def run_test() -> dict:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = ProxyClient(
                "http://proxy.test/",
                client=http_client,
                default_headers={
                    "Authorization": "Bearer proxy-secret",
                    "X-Client": "harbor",
                },
            )
            await client.chat_completions(
                {"model": "test", "messages": []},
                session_id="session-1",
                instance_id="instance-1",
            )
            return await client.capabilities()

    capabilities = asyncio.run(run_test())

    assert capabilities["current_weight_version"] == "weights-v7"
    assert [request.url.path for request in requests] == [
        "/v1/chat/completions",
        "/integration/capabilities",
    ]
    for request in requests:
        assert request.headers["authorization"] == "Bearer proxy-secret"
        assert request.headers["x-client"] == "harbor"
    assert requests[0].headers["x-session-id"] == "session-1"
    assert requests[0].headers["x-instance-id"] == "instance-1"


def test_proxy_client_default_timeout_is_bounded():
    async def run_test() -> tuple[float | None, float | None]:
        client = ProxyClient("http://proxy.test")
        try:
            return client._client.timeout.connect, client._client.timeout.read
        finally:
            await client.close()

    connect, read = asyncio.run(run_test())

    assert connect == 10.0
    assert read == 300.0
