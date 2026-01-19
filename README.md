# PixelHolo Trial (Monorepo)

Corporate-style layout with three project repos under a single top-level workspace.

## Layout
- `apps/frontend/` — UI control panel (Vite/React)
- `services/voice_cloning/` — StyleTTS2 training + inference
- `services/lip_syncing/` — Wav2Lip chunked + single-pass runners
- `reference/voxclone-control-panel/` — old UI reference (not used in production)

## Quick Start (dev)
1) Backend (voice + lips)
```bash
cd /home/alvin/PixelHolo_trial/services/voice_cloning
# voice only
python src/speak.py --profile <name> --text "Hello"
# voice + lips (chunked)
python src/speak_video.py --profile <name> --text "Hello"
```

2) Frontend
```bash
cd /home/alvin/PixelHolo_trial/apps/frontend
npm install
npm run dev
```

## Notes
- Outputs are organized by type:
  - `services/voice_cloning/outputs/audio/<profile>/`
  - `services/voice_cloning/outputs/video/<profile>/`
- The lip sync repo is a sibling under `services/` so the default `speak_video.py` path works.
