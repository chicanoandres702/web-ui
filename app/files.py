import os
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.base_router import BaseRouter
import logging

logger = logging.getLogger(__name__)

class FilesRouter(BaseRouter):
    def register_routes(self):
        @self.router.get("/files")
        async def list_files():
            files_list = []
            for root, dirs, files in os.walk("./tmp/knowledge_base"):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, ".").replace("\\", "/")
                    files_list.append(rel_path)
            return files_list

        @self.router.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            upload_dir = Path("./tmp/knowledge_base")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return {"filename": file.filename, "path": str(file_path)}
