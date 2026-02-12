from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import uuid
from datetime import datetime

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    WAITING_FOR_USER = "waiting_for_user"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: str = "pending"

class AgentTaskRequest(BaseModel):
    task: str
    class_name: str = "General"
    require_confirmation: bool = True
    model_override: Optional[str] = None
    headless: bool = False
    stream_quality: int = 50

class AgentState(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_input: str
    status: TaskStatus = TaskStatus.QUEUED
    plan: List[SubTask] = []
    result: Optional[str] = None
    error: Optional[str] = None
    class_name: str = "General"
    timestamp: datetime = Field(default_factory=datetime.now)

class UserFeedback(BaseModel):
    task_id: str
    approved: bool
    modified_plan: Optional[List[SubTask]] = None
