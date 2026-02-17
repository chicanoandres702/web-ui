import uvicorn
import sys, os
import asyncio
import nest_asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_server():
    nest_asyncio.apply()
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("app.core.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)

if __name__ == "__main__":
    run_server()
