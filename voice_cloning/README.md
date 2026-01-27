# Voice Cloning and Orchestration (StyleTTS2)

This module handles voice cloning (StyleTTS2) and orchestrates the full pipeline. It calls `lip_syncing/` to generate video frames during streaming.

## Key Concepts

### Avatar Baking
Real-time face detection is too slow. Baking happens once during preprocessing
1) Extract a short loop from the video.
2) Run face detection on each frame offline.
3) Save `frames.npy` and `coords.npy`.
4) Runtime loads the cache instantly.

### Staircase Chunking
Streaming uses a text pattern to prevent stalls:
- Chunk 1: 4 words (warmup).
- Chunk 2: 10 words (bridge).
- Chunk 3+: 25 words (cruise).

## Workflow

Choose a mode:
- Voice-only: audio output only.
- Voice + Avatar: audio + lip-synced video.

### 1a) Preprocess (voice-only)
```bash
python src/preprocess.py \
  --video /path/to/me.mp4 \
  --name alvin
```

### 1b) Preprocess (video -> dataset + avatar cache)
```bash
python src/preprocess_video.py \
  --video /path/to/me.mp4 \
  --name alvin
```

### 2) Train (recommended sprint)
```bash
python src/train.py \
  --dataset_path data/avatar_profiles/alvin \
  --profile_type avatar \
  --epochs 25
```

### 3) Streaming API
```bash
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```
Endpoints:
- `POST /stream_avatar` - NDJSON stream (audio + JPEG frames)
- `POST /stream` - audio-only stream

### 4) CLI video output
```bash
python src/speak_video.py \
  --profile alvin \
  --text "This is a generated video using the baked cache."
```

### Voice-only inference (audio)
```bash
python src/speak.py \
  --profile alvin \
  --profile_type voice \
  --text "Hello from voice-only."
```

## Configuration
Files of interest:
- `outputs/training/avatar/<name>/profile.json` - inference defaults (model path, ref wav, alpha/beta, f0_scale).
- `outputs/training/avatar/<name>/best_epoch.txt` - selected checkpoint.
- `outputs/training/avatar/<name>/epoch_scores.json` - scoring history.

Suggested training settings:
- `epochs: 25`
- `save_every: 1` to capture early sweet-spot checkpoints.

## Setup
```bash
cd /home/alvin/PixelHolo_trial/voice_cloning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
If `python` is not available, use `python3 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.
If you see an "externally-managed-environment" error, use `.venv/bin/python -m pip install -r requirements.txt`.

System deps: `ffmpeg`, `espeak-ng`.

Clone StyleTTS2 and download LibriTTS weights:
```bash
mkdir -p lib
cd lib
git clone https://github.com/yl4579/StyleTTS2.git
mkdir -p StyleTTS2/Models/LibriTTS
wget -O StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth
wget -O StyleTTS2/Models/LibriTTS/config.yml   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml
```

## Troubleshooting
- "Command not found": activate `.venv` first.
- "Lip sync bridge failed": `lip_syncing/` must be a sibling of `voice_cloning/`.
- "Stream stalls": check GPU VRAM usage and network stability.
- Training crashes on tiny datasets:
  - `ValueError: high <= 0` means too few segments (use a longer clip or `--legacy_split`).
  - `IndexError: Dimension out of range` can happen with `batch_size=1` on tiny datasets; use `--batch_size 2` for smoke tests.
