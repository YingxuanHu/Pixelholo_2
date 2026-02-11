# PixelHolo Voice Studio (Frontend)

This frontend is the control panel for the local StyleTTS2 workflow.
It talks to the FastAPI backend in `voice_cloning`.

## Run

1) Start the backend:

```bash
cd /home/alvin/PixelHolo_trial/voice_cloning
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

2) Install frontend deps + start:

```bash
cd /home/alvin/PixelHolo_trial/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open from desktop:
- `http://127.0.0.1:5173`

Open from iPhone (same network):
- `http://<vm-or-host-ip>:5173`

The app auto-detects backend as `http://<current-host>:8000`.
You can still override API Base in the header.

## Optional API Override

If frontend and backend are on different hosts, create:

```bash
frontend/.env.local
```

with:

```bash
VITE_API_BASE=http://<backend-ip>:8000
```

## iOS Notes

- Use Safari for best compatibility.
- Add to Home Screen works via web manifest (`standalone` display).
- Input controls use mobile-safe font sizing to avoid iOS auto-zoom.

## Backend Endpoints Used

- `POST /upload` (multipart)
- `POST /preprocess` (stream logs)
- `POST /train` (stream logs)
- `POST /stream` (NDJSON streaming audio chunks)
- `POST /stream_avatar` (NDJSON streaming audio + frames)
- `POST /chat` (LLM + TTS stream)
- `POST /speak` (TTS stream)
- `GET /profiles` (list existing profiles)

The UI plays audio as soon as the first chunk arrives.
