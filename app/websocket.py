from typing import Any
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self): 
        self.conns: list[WebSocket] = []

    async def connect(self, ws: WebSocket): 
        await ws.accept()
        self.conns.append(ws)

    def disconnect(self, ws: WebSocket): 
        self.conns.remove(ws)

    async def broadcast(self, type_: str, content: Any):
        for c in self.conns:
            try: 
                await c.send_json({"type": type_, "content": content})
            except Exception: 
                pass

manager = ConnectionManager()