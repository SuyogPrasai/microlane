from fastapi import FastAPI
from microlane.models.ufld.ufld.evaluate import UFLD
from microlane.models.ufld.ufld.routers.inference import router

app = FastAPI(title="UFLD")
app.include_router(router)


@app.on_event("startup")
def startup():
    app.state.model = UFLD(weights_path="weights/tusimple_18.pth")


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