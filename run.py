import uvicorn
import sys, os
import asyncio

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    print("Starting Scholar Agent Pro...")
    uvicorn.run("app.backend:create_app", host="0.0.0.0", port=8000, reload=True, factory=True)