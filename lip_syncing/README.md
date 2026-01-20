# Lip Syncing (Wav2Lip)

Chunked and single-pass Wav2Lip runners for PixelHolo. This repo can be used standalone or driven by `voice_cloning/src/speak_video.py`.

## TL;DR Quickstart
```bash
cd /home/alvin/PixelHolo_trial/lip_syncing
source .venv/bin/activate
python src/run_lipsync.py --video /path/to/source_video.mp4
```

## Layout
- `lib/Wav2Lip/` - Wav2Lip repo (clone here)
- `models/` - Wav2Lip checkpoints
- `src/chunked_lipsync.py` - chunked runner
- `src/run_lipsync.py` - wrapper (extract audio + run chunked)
- `outputs/` - generated results

## Requirements
- Python 3.12
- CUDA GPU recommended
- `ffmpeg` installed

## Setup
```bash
cd /home/alvin/PixelHolo_trial/lip_syncing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install GPU Torch (RTX 5090 / CUDA 12.8):
```bash
pip install --upgrade pip
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128   --index-url https://download.pytorch.org/whl/cu128
```

Clone Wav2Lip:
```bash
git clone https://github.com/Rudrabha/Wav2Lip.git lib/Wav2Lip
```

## Model downloads
Place these files in `lip_syncing/models/`:
- `s3fd-619a316812.pth`
  - https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
- `wav2lip_gan.pth`
  - https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k

Expected folder:
```
lip_syncing/
  models/
    s3fd-619a316812.pth
    wav2lip_gan.pth
```

## Run (chunked)
```bash
python src/chunked_lipsync.py   --video /path/to/source_video.mp4   --audio /path/to/generated_audio.wav   --output outputs/lipsync_out.mp4   --chunk_sec 1.0   --loop_sec 10   --fps 25   --resize_factor 2
```

## Run (wrapper)
Extracts audio from the input video, runs chunked lipsync, and writes `outputs/<video>_lipsync.mp4`.
```bash
python src/run_lipsync.py   --video /path/to/source_video.mp4   --resize_factor 2
```

## Run from voice_cloning
`voice_cloning/src/speak_video.py` calls this repo automatically. Example:
```bash
python /home/alvin/PixelHolo_trial/voice_cloning/src/speak_video.py   --profile alice   --text "Hello"
```

## Quality tips
- Sharper output: `--resize_factor 1` (heavier GPU)
- If you hit OOM: `--face_det_batch_size 4 --wav2lip_batch_size 64`
- Adjust mouth crop: `--pads "0 20 0 0"`
- Higher quality intermediate frames: `--fourcc MJPG`
- For best final quality: `--concat_mode copy`
- Preserve input quality during looping/chunking:
  `--loop_crf 18 --loop_preset slow --chunk_crf 18 --chunk_preset slow`
- If source is HDR (iPhone HLG), tone-mapping is applied automatically in the wrapper.

## Notes
- Wav2Lip expects 16 kHz mono audio; the script resamples per chunk.
- Chunking is for low-latency testing and streaming.


## Troubleshooting
- **No face detected**: try `--pads` and `--resize_factor 1`.
- **OOM**: lower `--face_det_batch_size` or `--wav2lip_batch_size`.
- **Bad colors (HDR)**: run through the wrapper (`run_lipsync.py`) which tone-maps.
