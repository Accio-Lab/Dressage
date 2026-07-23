from __future__ import annotations

from fastapi import APIRouter, Request

from blackbox_server import __version__
from blackbox_server.core.models import StatusResponse


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, object]:
    return {"healthy": True, "version": __version__}


@router.get("/v1/status", response_model=StatusResponse)
async def status(request: Request) -> StatusResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.status()
    response.request_id = request.state.request_id
    return response
