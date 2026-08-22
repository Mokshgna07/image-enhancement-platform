from fastapi import FastAPI

app = FastAPI(
    title="Image Enhancement API",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "image-enhancement-api",
    }
