# Realtime Pipeline

This package ingests live sensor readings from ThingSpeak, imputes dissolved
oxygen when the probe is offline, runs every trained model, and publishes the
results for the dashboard.

## Components

- `config.py` – loads environment-driven configuration (channel ID, field
  mapping, polling interval, dashboard webhook URL).
- `thing_speak_client.py` – lightweight HTTP client for ThingSpeak.
- `normalizer.py` – maps raw ThingSpeak fields to the features required by the
  models and estimates TDS from conductivity.
- `model_service.py` – loads all models once and serves aggregated predictions;
  degradation forecasting automatically enables when TensorFlow is installed.
- `pipeline.py` – orchestrates ingestion, inference, JSON persistence, and
  optional webhook publishing.
- `run_service.py` – CLI entry point (`python -m ml_backend.realtime.run_service`).

## Configuration

Set the following environment variables (e.g. via `.env`):

```
THINGSPEAK_CHANNEL_ID=YOUR_CHANNEL
THINGSPEAK_READ_API_KEY=XXXXXXX        # optional if channel is public
THINGSPEAK_FIELD_MAP={"temperature":"field1","ph":"field2","conductivity":"field3","turbidity":"field4","dissolved_oxygen":"field5"}
POLL_INTERVAL_SECONDS=60
REALTIME_OUTPUT_PATH=ml_backend/realtime/latest_output.json
DASHBOARD_WEBHOOK_URL=https://dashboard.example.com/api/water-quality    # optional
```

### Enabling the degradation forecast

Install TensorFlow in the runtime environment (CPU build shown):

```bash
pip install tensorflow==2.15.0
```

Once installed, restart the realtime service—the LSTM forecast will activate
automatically and the dashboard will begin showing the forecast timeline.

## Deploying on Render

The repository ships with a `render.yaml` blueprint describing two services:

- **Web service** (`neerwana-realtime-api`): runs the FastAPI app (`ml_backend/realtime/app.py`) and
  serves `/api/latest`, `/dashboard/latest.json`, and the static `frontend/` assets.
- **Background worker** (`neerwana-realtime-worker`): runs the ingestion loop (`python -m ml_backend.realtime.run_worker`)
  so predictions stay fresh.

### Steps

1. Push the code to GitHub.
2. In Render, create a new Blueprint deployment pointing at the repository (Render automatically picks up `render.yaml`).
3. Add the environment variables listed above to both the web service and the worker (channel ID, API key, field map, etc.).
4. Upgrade the worker to a paid plan if you need 24/7 polling (free tier workers may sleep when idle).

Once deployed:

- `https://<web-service>.onrender.com/api/latest` returns the latest prediction packet.
- `https://<web-service>.onrender.com/dashboard/latest.json` exposes the same JSON file.
- `https://<web-service>.onrender.com/` serves the dashboard UI.

Run once:

```
python -m ml_backend.realtime.run_service
```

Run continuously:

```
python -m ml_backend.realtime.run_service --loop
```

When a new reading arrives the pipeline writes a consolidated JSON payload to
`REALTIME_OUTPUT_PATH` and POSTs the same payload to `DASHBOARD_WEBHOOK_URL`
(if provided). The payload contains:

- Normalised sensor values with a `DO_imputed` flag
- WQI score and pollution class
- Contamination cause prediction
- Disease outbreak risk
- Degradation forecast or an explanation if unavailable (TensorFlow missing or
  insufficient history)

Use the JSON file or the webhook to drive the web dashboard. The same pipeline
keeps a short history buffer that satisfies the LSTM model once TensorFlow is
installed.


