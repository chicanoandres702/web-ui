import os
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile

router = APIRouter()

@router.get("/files")
async def list_files():
    files_list = []
    for root, dirs, files in os.walk("./tmp"):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ".").replace("\\", "/")
            files_list.append(rel_path)
    return files_list

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = Path("./tmp/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": str(file_path)}
