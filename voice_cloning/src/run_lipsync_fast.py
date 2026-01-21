import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILE_TYPE_AVATAR
from src.lipsync_bridge import LipSyncBridge


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Command failed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast Wav2Lip runner using cached avatar frames/coords."
    )
    parser.add_argument("--profile", required=True, help="Avatar profile name.")
    parser.add_argument(
        "--profile_type",
        default=PROFILE_TYPE_AVATAR,
        help="Profile type (avatar).",
    )
    parser.add_argument("--audio", required=True, type=Path, help="Input audio wav.")
    parser.add_argument("--output", required=True, type=Path, help="Output mp4.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Wav2Lip checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--wav2lip_batch_size", type=int, default=128)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="slow")

    args = parser.parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio}")

    t0 = time.perf_counter()
    engine = LipSyncBridge(
        checkpoint_path=args.checkpoint,
        device=args.device,
        wav2lip_batch_size=args.wav2lip_batch_size,
    )
    engine.load_profile(args.profile, args.profile_type)

    audio_16k, _ = librosa.load(str(args.audio), sr=16000, mono=True)
    frames = engine.sync_chunk(audio_16k)
    if not frames:
        raise RuntimeError("No frames generated from Wav2Lip.")

    fps = engine.fps or 25.0
    height, width = frames[0].shape[:2]
    tmp_dir = Path(tempfile.mkdtemp(prefix="lipsync_fast_"))
    tmp_avi = tmp_dir / "result.avi"

    writer = cv2.VideoWriter(
        str(tmp_avi),
        cv2.VideoWriter_fourcc(*args.fourcc),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_avi),
        "-i",
        str(args.audio),
        "-c:v",
        "libx264",
        "-crf",
        str(args.crf),
        "-preset",
        args.preset,
        "-c:a",
        "aac",
        "-shortest",
        str(args.output),
    ]
    _run(cmd)
    print(f"[done] Saved: {args.output} ({time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    main()
