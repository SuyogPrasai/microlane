from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from microlane.models.ufld.ufld.schemas.requests import InferRequest
from microlane.models.ufld.ufld.helpers.server_utils import prediction_to_dict, sample_request_to_dict

router = APIRouter()

@router.post("/infer")
def infer(request: InferRequest, app_request: Request):
    try:
        sample = request.sample.to_sample()
        
        prediction = app_request.app.state.model.infer(sample)
        
        response = prediction_to_dict(prediction)
        
        return JSONResponse(content=response)
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))