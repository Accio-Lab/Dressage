from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from blackbox_server.core.errors import ApiError
from blackbox_server.core.models import (
    AbortResponse,
    ExecuteCmdRequest,
    ExecuteCmdResponse,
    MessageRequest,
    SessionResponse,
    TurnCancelResponse,
    TurnStatusResponse,
    TurnSubmitResponse,
)


router = APIRouter()


@router.post("/v1/sessions/{session_id}/messages")
async def send_message(session_id: str, request: Request, payload: MessageRequest):
    server = request.app.state.server
    if payload.mode == "async":
        if payload.turn_id is None:
            raise ApiError(
                400,
                "request_error",
                "turn_id is required for async submissions.",
                details={"session_id": session_id},
            )
        async with server.request_scope():
            submission = await server.submit_turn(session_id, payload)
        response = TurnSubmitResponse(
            request_id=request.state.request_id,
            session_id=session_id,
            instance_id=submission.instance_id,
            turn_id=submission.turn_id,
            status=submission.status,
            idempotent_replay=submission.idempotent_replay,
        )
        return JSONResponse(status_code=202, content=response.model_dump(mode="json"))

    async with server.request_scope():
        response = await server.send_message(session_id, payload)
    response.request_id = request.state.request_id
    return response


@router.get(
    "/v1/sessions/{session_id}/turns/{turn_id}",
    response_model=TurnStatusResponse,
)
async def get_turn(
    session_id: str,
    turn_id: str,
    request: Request,
    wait: float = Query(default=0.0, ge=0.0),
) -> TurnStatusResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.get_turn(session_id, turn_id, wait_seconds=wait)
    response.request_id = request.state.request_id
    return response


@router.post(
    "/v1/sessions/{session_id}/turns/{turn_id}/cancel",
    response_model=TurnCancelResponse,
)
async def cancel_turn(session_id: str, turn_id: str, request: Request) -> TurnCancelResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.cancel_turn(session_id, turn_id)
    response.request_id = request.state.request_id
    return response


@router.post("/v1/sessions/{session_id}/execute_cmd", response_model=ExecuteCmdResponse)
async def execute_cmd(session_id: str, request: Request, payload: ExecuteCmdRequest) -> ExecuteCmdResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.execute_cmd(session_id, payload)
    response.request_id = request.state.request_id
    return response


@router.get("/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    request: Request,
    include_history: bool = Query(default=False),
    include_trace: bool = Query(default=False),
    include_turns: bool = Query(default=False),
) -> SessionResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.get_session(
            session_id,
            include_history=include_history,
            include_trace=include_trace,
            include_turns=include_turns,
        )
    response.request_id = request.state.request_id
    return response


@router.post("/v1/sessions/{session_id}/abort", response_model=AbortResponse)
async def abort_session(session_id: str, request: Request) -> AbortResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.abort_session(session_id)
    response.request_id = request.state.request_id
    return response
