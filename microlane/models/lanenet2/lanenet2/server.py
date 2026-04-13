from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from schema.api_schemas import Sample
from schema.api_schemas import LaneNet2Output
from evaluate import LaneNet2

class InferRequest(BaseModel):
    sample: Sample

    class Config:
        arbitrary_types_allowed = True


class BatchInferRequest(BaseModel):
    samples: List[Sample]

    class Config:
        arbitrary_types_allowed = True


def prediction_to_dict(pred: LaneNet2Output) -> dict:
    result = {}
    for key, val in pred.__dict__.items():
        result[key] = val.tolist() if isinstance(val, np.ndarray) else val
    return result


app = FastAPI(title="LaneNet2 API")


@app.on_event("startup")
def startup():
    global model
    model = LaneNet2(weights_path="weights/tusimple_lanenet.ckpt")


@app.on_event("shutdown")
def shutdown():
    model.close()
    


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
def infer(request: InferRequest):
    try:
        prediction = model.infer(request.sample) # type: ignore
        return JSONResponse(content=prediction_to_dict(prediction)) # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_infer")
def batch_infer(request: BatchInferRequest):
    try:
        predictions = model.batch_infer(request.samples) # type: ignore
        return JSONResponse(content=[prediction_to_dict(p) for p in predictions]) # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)