import tempfile

import numpy as np
import pandas as pd

from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError, validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

app = FastAPI(docs_url="/docs")

# model: Model = Depends(get_model)
@app.post("/predict")
def predict(img: bytes = File(...)):
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as f:
        f.write(img)
        return FileResponse(f.name, media_type="image/png")