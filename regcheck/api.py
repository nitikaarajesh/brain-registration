"""RegCheck web API: POST two NIfTI files, get the verdict back as JSON."""
import os, shutil, tempfile
from fastapi import FastAPI, UploadFile, File
from regcheck import regcheck

app = FastAPI(title="RegCheck")

@app.post("/register")
async def register(fixed: UploadFile = File(...), moving: UploadFile = File(...)):
    tmp = tempfile.mkdtemp()
    paths = {}
    for name, up in [("fixed", fixed), ("moving", moving)]:
        paths[name] = os.path.join(tmp, f"{name}.nii.gz")
        with open(paths[name], "wb") as f:
            shutil.copyfileobj(up.file, f)
    report = regcheck(paths["fixed"], paths["moving"], os.path.join(tmp, "out"))
    shutil.rmtree(tmp, ignore_errors=True)
    return report