# Voice Cloning (StyleTTS2)

Local StyleTTS2 fine-tuning plus FastAPI streaming inference. This service also drives lip sync by calling the sibling `lip_syncing/` repo when requested.

## TL;DR Quickstart
```bash
cd /home/alvin/PixelHolo_trial/voice_cloning
source .venv/bin/activate
python src/preprocess.py --video /path/to/user.mp4 --name alice --profile_type voice
python src/train.py --dataset_path ./data/voice_profiles/alice --profile_type voice
python src/speak.py --profile alice --text "Hello"
python src/speak_video.py --profile alice --text "Hello from video"
```

## Requirements
- System deps: `ffmpeg`, `espeak-ng`
- Python 3.12
- NVIDIA GPU recommended

## Setup
```bash
cd /home/alvin/PixelHolo_trial/voice_cloning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
If `python` isn't available on your system, use `python3 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.

Clone StyleTTS2 and download LibriTTS weights:
```bash
mkdir -p lib
cd lib
git clone https://github.com/yl4579/StyleTTS2.git
mkdir -p StyleTTS2/Models/LibriTTS
wget -O StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth
wget -O StyleTTS2/Models/LibriTTS/config.yml   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml
```

If `faster-whisper` fails to load cuDNN from pip packages:
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cudnn/lib:$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cublas/lib
```

## Project layout
- `data/`
  - `voice_profiles/<profile>/` - audio-only dataset
  - `avatar_profiles/<profile>/` - video-enabled dataset (includes `avatar_cache/`)
- `outputs/`
  - `training/voice/<profile>/` - checkpoints + `config_ft.yml` + `profile.json`
  - `training/avatar/<profile>/` - same but for avatar profiles
  - `audio/voice/<profile>/` - generated wav (speak.py, auto-select samples)
  - `audio/avatar/<profile>/` - generated wav for avatar runs
  - `video/avatar/<profile>/` - generated lip-sync mp4
- `src/` - core scripts
- `config.py` - shared settings

Key files:
- `profile.json` - inference defaults (model path, ref wav, alpha/beta, f0_scale).
- `best_epoch.txt` - selected checkpoint from auto-select.
- `epoch_scores.json` - scoring history for all checkpoints.
- `f0_scale.txt` - auto-estimated pitch scale.

## Workflow (end-to-end)

### 1) Preprocess
Splits the input video into clean clips, transcribes them, and writes `metadata.csv`. It also bakes the avatar cache used by lip sync.
```bash
python src/preprocess.py --video /path/to/user.mp4 --name alice --profile_type voice
```
Notes:
- Default preprocessing uses HP/LP filters and stricter text filtering.
- Denoise is optional (`--denoise`).
- For avatar profiles, use `src/preprocess_video.py` instead.

### 1b) Preprocess (avatar video)
Builds the audio dataset and avatar cache in one pass:
```bash
python src/preprocess_video.py --video /path/to/user.mp4 --name alice
```

### 2) Train
```bash
python src/train.py --dataset_path ./data/voice_profiles/alice --profile_type voice
```
Common overrides:
```bash
python src/train.py --dataset_path ./data/voice_profiles/alice --profile_type voice   --batch_size 2 --max_len 200
```
Optional auto tools:
```bash
python src/train.py --dataset_path ./data/voice_profiles/alice --profile_type voice   --auto_tune_profile   --auto_select_epoch   --auto_build_lexicon
```

### 3) Inference (audio)
```bash
python src/speak.py --profile alice --profile_type voice --text "Hello world"
```

### 4) Inference (voice + lip sync)
Requires the sibling repo:
```
/home/alvin/PixelHolo_trial/lip_syncing
```
and the Wav2Lip models in `lip_syncing/models/`.

```bash
python src/speak_video.py --profile alice --text "Hello from video"
```

## API server
```bash
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /stream` - audio only (NDJSON)
- `POST /stream_avatar` - audio + JPEG frames (NDJSON)
- `POST /generate` - full WAV

## Flag reference (common)

### Preprocess (`src/preprocess.py`)
- `--video` input path
- `--name` profile name (creates `data/<name>/`)
- `--language` Whisper language (default `en`)
- `--denoise` optional
- `--min_words`, `--min_speech_ratio`, `--min_avg_logprob`, `--max_no_speech_prob`

### Preprocess (avatar) (`src/preprocess_video.py`)
- All audio flags above, plus:
- `--no_bake_avatar` to skip lip sync cache
- `--avatar_fps`, `--avatar_loop_sec`, `--avatar_loop_fade_sec`, `--avatar_resize_factor`, `--avatar_pads`
- `--avatar_batch_size`, `--avatar_nosmooth`, `--avatar_device`

### Train (`src/train.py`)
- `--dataset_path` input dataset
- `--output_dir` override training output
- `--epochs`, `--batch_size`, `--max_len`, `--grad_accum_steps`
- `--max_text_chars`, `--max_text_words` to avoid BERT limits
- `--auto_tune_profile`, `--auto_select_epoch`, `--auto_build_lexicon`
- `--tune_ref_wav`, `--select_ref_wav` to force a reference
- `--tune_thorough`, `--select_thorough` for multi-ref scoring
- `--tune_quick`, `--select_quick` for a fast pass
- `--lexicon_lang` to build a per-profile lexicon

### Inference (`src/speak.py`)
- `--profile`, `--text`
- `--ref_wav` override
- `--phonemizer_lang` accent (e.g., `en-ca`, `en-us`, `en-gb`)
- `--max_chunk_chars`, `--max_chunk_words`, `--pause_ms`
- `--pitch_shift`, `--de_esser_cutoff`, `--de_esser_order`
- `--seed` for deterministic output

### Video inference (`src/speak_video.py`)
- `--profile`, `--text`
- `--video` optional override
- `--lipsync_dir` path to `lip_syncing` (default sibling)
- `--lipsync_python` optional venv python
- `--lipsync_mode` (`chunked`, `lib`, `cached`)
- Chunked options: `--chunk_sec`, `--resize_factor`, `--fourcc`, `--face_det_batch_size`, `--wav2lip_batch_size`, `--pads`, `--concat_mode`

## Notes
- `profile.json` is stored next to the chosen checkpoint and is used by `speak.py`.
- If `profile.json` exists, it provides default model path, ref wav, and tuning parameters.
- `auto_select_epoch` writes `best_epoch.txt` and `epoch_scores.json` in the training folder.
- If no ref wav is provided, the script auto-picks a clean clip from `processed_wavs`.


## Troubleshooting
- **CUDA/cuDNN load errors**: install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`, then export `LD_LIBRARY_PATH` as described above.
- **BERT length errors**: lower `--max_text_chars` or `--max_text_words`.
- **OOM**: reduce `--batch_size` and `--max_len`.
- **Lip sync missing**: ensure `lip_syncing/` exists as a sibling and models are in `lip_syncing/models/`.
- **uvicorn not executable**: reinstall in the venv (`python -m pip install --force-reinstall uvicorn`) or run `python -m uvicorn ...`.
