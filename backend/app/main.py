from fastapi import FastAPI

app = FastAPI(title="RecoveryOS")


@app.get("/health")
def health():
    return {"status": "ok"}
