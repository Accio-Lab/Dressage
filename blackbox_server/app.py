from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from blackbox_server import __version__
from blackbox_server.api import health_router, rollout_router, sessions_router
from blackbox_server.config import BlackboxServerConfig
from blackbox_server.core.errors import ApiError
from blackbox_server.core.models import ErrorResponse
from blackbox_server.core.server import BlackboxServer


LOGGER = logging.getLogger("blackbox_server.http")


def _format_body_for_log(body: bytes) -> str:
    if not body:
        return "-"
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text.replace("\n", "\\n")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    server = app.state.server
    yield
    await server.graceful_shutdown()


def create_app(config: BlackboxServerConfig | None = None) -> FastAPI:
    app = FastAPI(title="BlackboxServer", version=__version__, lifespan=lifespan)
    app.state.server = BlackboxServer(config or BlackboxServerConfig.from_env())

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = f"req-{uuid4().hex[:8]}"
        started_at = perf_counter()
        client_host = request.client.host if request.client is not None else "-"
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        request_body = await request.body()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1000
            LOGGER.exception(
                "request failed request_id=%s method=%s path=%s duration_ms=%.1f client=%s request_body=%s",
                request.state.request_id,
                request.method,
                path,
                duration_ms,
                client_host,
                _format_body_for_log(request_body),
            )
            raise
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk if isinstance(chunk, bytes) else chunk.encode()
        duration_ms = (perf_counter() - started_at) * 1000
        response.headers["x-request-id"] = request.state.request_id
        LOGGER.info(
            (
                "request completed request_id=%s method=%s path=%s status_code=%d "
                "duration_ms=%.1f client=%s request_body=%s response_body=%s"
            ),
            request.state.request_id,
            request.method,
            path,
            response.status_code,
            duration_ms,
            client_host,
            _format_body_for_log(request_body),
            _format_body_for_log(response_body),
        )
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        payload = ErrorResponse(
            request_id=getattr(request.state, "request_id", f"req-{uuid4().hex[:8]}"),
            error=exc.error,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(
            request_id=getattr(request.state, "request_id", f"req-{uuid4().hex[:8]}"),
            error="request_error",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=400, content=payload.model_dump(mode="json"))

    app.include_router(health_router)
    app.include_router(rollout_router)
    app.include_router(sessions_router)
    return app
