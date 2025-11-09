# Water Quality Monitoring Platform

End-to-end system for real-time water quality assessment, combining IoT ingestion,
machine learning models, and a responsive dashboard.

## Components

- **ml_backend/**  
  - `realtime/` – ThingSpeak ingestion, DO imputation, model orchestration, FastAPI service, Render deployment assets.  
  - Domain models for WQI classification, contamination detection, disease outbreak prediction, and degradation forecasting.
- **frontend/** – Responsive dashboard that polls the realtime JSON feed and highlights safety status, contamination alerts, disease risk, and forecasts.
- **tests/** – Smoke tests for the realtime pipeline.

## Quick Start

```bash
python -m ml_backend.realtime.run_service --loop    # start ingestion loop
uvicorn ml_backend.realtime.app:app --reload        # serve API + dashboard
```

Then open `http://localhost:8000/frontend/index.html`.

## Deployment

Render blueprint (`render.yaml`) provisions:
- `neerwana-realtime-api` (FastAPI web service + dashboard)
- `neerwana-realtime-worker` (ThingSpeak polling worker)

Configure environment variables (`THINGSPEAK_CHANNEL_ID`, `THINGSPEAK_FIELD_MAP`, etc.) for both services. TensorFlow 2.15.0 is required for degradation forecasting.

