# PixelHolo Trial (Monorepo)

PixelHolo Trial bundles three tools under one workspace:
- Voice cloning (StyleTTS2)
- Lip syncing (Wav2Lip)
- Frontend control panel (React)

## Layout
- `frontend/` - UI control panel (Vite/React)
- `voice_cloning/` - StyleTTS2 training + inference + API
- `lip_syncing/` - Wav2Lip chunked and single-pass runners
- `reference/` - legacy UI reference (not used in production)

## Requirements
- Linux + NVIDIA GPU recommended
- `ffmpeg` and `espeak-ng` installed system-wide
- Python 3.12 for `voice_cloning/`
- Node 18+ for `frontend/`

## One-time setup
1) Voice cloning
```bash
cd /home/alvin/PixelHolo_trial/voice_cloning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Clone StyleTTS2 and download LibriTTS weights:
```bash
mkdir -p lib
cd lib
git clone https://github.com/yl4579/StyleTTS2.git
mkdir -p StyleTTS2/Models/LibriTTS
wget -O StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth
wget -O StyleTTS2/Models/LibriTTS/config.yml   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml
```

2) Lip syncing
```bash
cd /home/alvin/PixelHolo_trial/lip_syncing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Clone Wav2Lip and download checkpoints:
```bash
git clone https://github.com/Rudrabha/Wav2Lip.git lib/Wav2Lip
# Place these files in lip_syncing/models/
# - s3fd-619a316812.pth
# - wav2lip_gan.pth
```

3) Frontend
```bash
cd /home/alvin/PixelHolo_trial/frontend
npm install
```

## Quick start (CLI)
1) Preprocess a video (builds voice dataset + avatar cache)
```bash
python /home/alvin/PixelHolo_trial/voice_cloning/src/preprocess.py   --video /path/to/user.mp4   --name alice
```

2) Train
```bash
python /home/alvin/PixelHolo_trial/voice_cloning/src/train.py   --dataset_path /home/alvin/PixelHolo_trial/voice_cloning/data/alice
```

3) Voice inference (audio)
```bash
python /home/alvin/PixelHolo_trial/voice_cloning/src/speak.py   --profile alice   --text "Hello world"
```

4) Voice + lip sync (video)
```bash
python /home/alvin/PixelHolo_trial/voice_cloning/src/speak_video.py   --profile alice   --text "Hello from video"
```

## Quick start (API + frontend)
Backend:
```bash
cd /home/alvin/PixelHolo_trial/voice_cloning
source .venv/bin/activate
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

Frontend:
```bash
cd /home/alvin/PixelHolo_trial/frontend
npm run dev
```

## Data and outputs
- `voice_cloning/data/<profile>/` - dataset + metadata
- `voice_cloning/outputs/training/<profile>/` - checkpoints + logs
- `voice_cloning/outputs/audio/<profile>/` - generated wav
- `voice_cloning/outputs/video/<profile>/` - generated mp4

## Notes
- `voice_cloning` will look for `lip_syncing` as a sibling folder under the repo root.
- `preprocess.py` bakes avatar frames for lip sync by default.
- `frontend` can run voice-only or voice+lip-sync using the same backend.
