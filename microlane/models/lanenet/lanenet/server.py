from fastapi import FastAPI # pyright: ignore[reportMissingImports]
from evaluate import LaneNet  # pyright: ignore[reportMissingImports]
from routers.inference import router  # pyright: ignore[reportMissingImports]

app = FastAPI(title="LaneNet")
app.include_router(router)


@app.on_event("startup")
def startup():
    app.state.model = LaneNet(weights_path="weights/tusimple_lanenet.ckpt")


@app.on_event("shutdown")
def shutdown():
    if app.state.model is not None:
        app.state.model.close()


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn  # pyright: ignore[reportMissingImports]
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)