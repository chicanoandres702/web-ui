from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Any

class ConnectionManager:
    def __init__(self): self.conns = []
    async def connect(self, ws: WebSocket): await ws.accept(); self.conns.append(ws)
    def disconnect(self, ws: WebSocket): self.conns.remove(ws)
    async def broadcast(self, type_: str, content: Any):
        for c in self.conns:
            try: await c.send_json({"type": type_, "content": content})
            except: pass

manager = ConnectionManager()
router = APIRouter()

@router.websocket("/ws/agent")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: manager.disconnect(ws)
