from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/events")
async def events(websocket: WebSocket):
    await websocket.accept()
    services = websocket.app.state.services
    while True:
        try:
            active = services.jobs.active()
            await websocket.send_json(
                {
                    "type": "status",
                    "queue_depth": services.jobs.queue_depth(),
                    "active_job": asdict(active) if active else None,
                    "nodes": [asdict(node) for node in services.nodes.list()],
                }
            )
            await asyncio.sleep(2)
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return
