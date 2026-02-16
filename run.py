import uvicorn
import sys
import asyncio
import nest_asyncio

if __name__ == "__main__":
    nest_asyncio.apply()
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)