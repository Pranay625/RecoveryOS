from fastapi import FastAPI

from app.routers.recovery import router as recovery_router

app = FastAPI(title="RecoveryOS")

app.include_router(recovery_router)


@app.get("/health")
def health():
    return {"status": "ok"}
