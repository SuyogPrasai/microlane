import traceback
import logging

from fastapi import APIRouter, HTTPException, Request  # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]

from schemas.requests import InferRequest
from helpers.request_conversion import samples_request_to_samples, prediction_to_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/infer")
def infer(request: InferRequest, app_request: Request):
    try:
        samples = samples_request_to_samples(request.samples)

        prediction = app_request.app.state.model.infer(samples)

        response = prediction_to_response(prediction)

        return JSONResponse(content=response)

    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))