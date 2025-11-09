"""
Simple FastAPI application that exposes the latest realtime predictions and static dashboard assets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import PipelineConfig, load_pipeline_config

app = FastAPI(title="NEERWANA Realtime API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_output_path(config: PipelineConfig) -> Path:
    output_path = Path(config.output_path)
    if not output_path.is_absolute():
        # Assume project root is two levels up (ml_backend/realtime/..)
        root = Path(__file__).resolve().parents[2]
        output_path = root / output_path
    return output_path


@app.get("/healthz")
def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/latest")
def get_latest_predictions() -> JSONResponse:
    config = load_pipeline_config()
    output_path = resolve_output_path(config)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Latest output not available yet.")
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Output file contains invalid JSON.") from exc
    return JSONResponse(payload)


@app.get("/dashboard/latest.json")
def get_dashboard_json() -> FileResponse:
    config = load_pipeline_config()
    output_path = resolve_output_path(config)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Latest output not available yet.")
    return FileResponse(output_path, media_type="application/json")


@app.get("/{path:path}")
def serve_frontend(path: Optional[str] = None):
    """
    Serve static frontend assets (index.html by default).
    """
    frontend_root = Path(__file__).resolve().parents[2] / "frontend"
    candidate = frontend_root / (path or "")

    if path and candidate.exists() and candidate.is_file():
        return FileResponse(candidate)

    index = frontend_root / "index.html"
    if index.exists():
        return FileResponse(index)

    raise HTTPException(status_code=404, detail="Asset not found.")

