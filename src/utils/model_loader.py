import importlib.util
import inspect
import logging
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def load_model_from_file(model_name: str):
    try:
        path = Path(f"./storage/{model_name}.py")
        if not path.exists():
            return None
        
        spec = importlib.util.spec_from_file_location(model_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                if obj.__module__ == model_name:
                    return obj
        return None
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None
