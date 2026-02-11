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
npm run dev
```

Open: `http://127.0.0.1:5173`

## iPhone / iOS Voice Input

iOS microphone APIs require a secure origin. If you open the app from your phone over plain LAN HTTP, Safari may block voice capture.

Recommended test flow:

1) Run backend and frontend locally.
2) Start an HTTPS tunnel to the frontend dev port:

```bash
ngrok http 5173
```

3) Open the `https://...ngrok-free.app` URL on your iPhone.
4) Update API Base in the app to a reachable backend URL (can be another ngrok URL for port 8000 if needed).

The `Voice Input` button uses:
- Native SpeechRecognition when available.
- Audio-only microphone capture + `/transcribe_audio` fallback when native speech recognition is unavailable.

## Backend Endpoints Used

- `POST /upload` (multipart)
- `POST /preprocess` (stream logs)
- `POST /train` (stream logs)
- `POST /stream` (NDJSON streaming audio chunks)
- `POST /transcribe_audio` (audio-to-text fallback)
- `GET /profiles` (list existing profiles)

The UI plays audio as soon as the first chunk arrives.
