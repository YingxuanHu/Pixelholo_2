# PixelHolo Trial (Monorepo)

Corporate-style layout with three project repos under a single top-level workspace.

## Layout
- `frontend/` — UI control panel (Vite/React)
- `voice_cloning/` — StyleTTS2 training + inference
- `lip_syncing/` — Wav2Lip chunked + single-pass runners
- `reference/voxclone-control-panel/` — old UI reference (not used in production)

## Quick Start (dev)
1) Backend (voice + lips)
```bash
cd /home/alvin/PixelHolo_trial/voice_cloning
# voice only
python src/speak.py --profile <name> --text "Hello"
# voice + lips (chunked)
python src/speak_video.py --profile <name> --text "Hello"
```

2) Frontend
```bash
cd /home/alvin/PixelHolo_trial/frontend
npm install
npm run dev
```

## Notes
- Outputs are organized by type:
  - `voice_cloning/outputs/audio/<profile>/`
  - `voice_cloning/outputs/video/<profile>/`
- The lip sync repo is a sibling under the monorepo root so the default `speak_video.py` path works.
