from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.task_orchestrator import TaskOrchestrator

@asynccontextmanager
async def lifespan(app: FastAPI):
    orc = TaskOrchestrator.get_instance()
    orc.start_worker()
    yield
    await orc.stop_worker()
