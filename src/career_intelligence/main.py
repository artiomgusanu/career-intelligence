
from fastapi import FastAPI

app = FastAPI(
    title="Career Intelligence",
    description="An intelligence platform for career and market research.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
