from fastapi import APIRouter
from abc import ABC, abstractmethod

class BaseRouter(ABC):
    def __init__(self):
        self.router = APIRouter()
        self.register_routes()

    @abstractmethod
    def register_routes(self):
        pass    
    def get_router(self) -> APIRouter:
        return self.router
