from fastapi import FastAPI
from microlane.models.lanenet2.lanenet2.evaluate import LaneNet2
from microlane.models.lanenet2.lanenet2.routers.inference import router

app = FastAPI(title="LaneNet2")
app.include_router(router)


@app.on_event("startup")
def startup():
    app.state.model = LaneNet2(weights_path="weights/tusimple_lanenet.ckpt")


@app.on_event("shutdown")
def shutdown():
    if app.state.model is not None:
        app.state.model.close()


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)