from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.recovery import router as recovery_router

app = FastAPI(title="RecoveryOS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(recovery_router)


@app.get("/health")
def health():
    return {"status": "ok"}
