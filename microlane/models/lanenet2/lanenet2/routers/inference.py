from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from schemas.requests import InferRequest, BatchInferRequest
from helpers.server_utils import prediction_to_dict, sample_request_to_dict

router = APIRouter()

@router.post("/infer")
def infer(request: InferRequest, app_request: Request):
    try:
        sample = request.sample.to_sample()
        prediction = app_request.app.state.model.infer(sample)
        response = prediction_to_dict(prediction)
        response["sample"] = sample_request_to_dict(request.sample)
        return JSONResponse(content=response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/batch_infer")
def batch_infer(request: BatchInferRequest, app_request: Request):
    try:
        samples = [s.to_sample() for s in request.samples]
        predictions = app_request.app.state.model.batch_infer(samples)
        return JSONResponse(content=[prediction_to_dict(pred) for pred in predictions])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))