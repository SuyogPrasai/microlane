from fastapi import APIRouter, HTTPException, Request # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse # pyright: ignore[reportMissingImports]

from schemas.requests import InferRequest
from helpers.request_conversion import sample_request_to_sample, prediction_to_response

router = APIRouter()

@router.post("/infer")
def infer(request: InferRequest, app_request: Request):
    try:
        sample = sample_request_to_sample(request.sample)
        
        prediction = app_request.app.state.model.infer(sample)
        
        response = prediction_to_response(prediction)
        
        return JSONResponse(content=response)
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))