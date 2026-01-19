# Lip Syncing (Chunked Wav2Lip CLI)

This repo adds a terminal-first lip-sync pipeline that works **in chunks**, similar to the voice_cloning flow.

## Layout
- `lib/Wav2Lip/` — Wav2Lip repo (clone here)
- `models/` — Wav2Lip checkpoints
- `outputs/` — generated chunks + final video
- `src/chunked_lipsync.py` — chunked CLI runner

## Setup
```bash
# 1) Create venv (optional)
python3.12 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Install GPU Torch (RTX 5090 / CUDA 12.8)
pip install --upgrade pip
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 4) Clone Wav2Lip
git clone https://github.com/Rudrabha/Wav2Lip.git lib/Wav2Lip
```

## GPU notes (RTX 5090)
Wav2Lip needs a modern PyTorch build for RTX 5090 (sm_120). Use the CUDA 12.8 wheels shown above.

## Model downloads
Download these files and place them in `lip_syncing/models/`:

- Face detection pre-trained model:
  - https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
- Wav2Lip GAN checkpoint:
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
python src/chunked_lipsync.py \
  --video /path/to/source_video.mp4 \
  --audio /path/to/generated_audio.wav \
  --output outputs/lipsync_out.mp4 \
  --chunk_sec 1.0 \
  --loop_sec 10 \
  --fps 25 \
  --resize_factor 2
```

## Run (one-shot wrapper)
Extracts audio from the input video, runs chunked lipsync, and writes `outputs/<video>_lipsync.mp4`.
```bash
python src/run_lipsync.py \
  --video /path/to/source_video.mp4 \
  --resize_factor 2
```

## Run (PixelHolo-style single-pass)
Uses the `lipsync` library (Wav2Lip wrapper) the same way PixelHolo does: load once,
cache face detections, and run a single pass over the full video.
```bash
python src/run_lipsync_lib.py \
  --video /path/to/source_video.mp4 \
  --output outputs/lipsync_out.mp4 \
  --checkpoint models/wav2lip_gan.pth \
  --cache_dir outputs/cache \
  --save_cache \
  --nosmooth
```
Notes:
- This path is **not chunked**. It trades latency for simpler, stable output.
- If your input is HDR, tone-mapping runs automatically. Disable with `--no_tonemap`.

Quality tips:
- For sharper output, try `--resize_factor 1` (heavier GPU).
- If you hit GPU OOM, use `--face_det_batch_size 4 --wav2lip_batch_size 64`.
- Tweak mouth crop with `--pads "0 20 0 0"` to include more chin/lip area.
- Use `--fourcc MJPG` for higher-quality intermediate frames.
- For best final quality, use `--concat_mode copy` (no re-encode). If that fails, switch to
  `--concat_mode reencode --concat_crf 18 --concat_preset slow`.
- Preserve input quality during looping/chunking with:
  `--loop_crf 18 --loop_preset slow --chunk_crf 18 --chunk_preset slow`.
- If your source is HDR (iPhone HLG), the wrapper will auto-tone-map to SDR.
  Disable with `--no_tonemap`.
- Control Wav2Lip output compression with:
  `--wav2lip_crf 18 --wav2lip_preset slow`.

## Notes
- Wav2Lip expects **16 kHz mono** audio. The script resamples per chunk.
- Chunking is for low-latency testing. It stitches small Wav2Lip outputs into one video.
- For best quality, start with a **loopable, stable head pose** video.
