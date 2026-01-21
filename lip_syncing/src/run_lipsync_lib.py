import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError("Command failed.")


def _ffprobe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=pix_fmt,color_transfer,color_primaries,color_space",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed.")
    return json.loads(result.stdout)


def _needs_tonemap(path: Path) -> bool:
    info = _ffprobe_video(path)
    streams = info.get("streams") or []
    if not streams:
        return False
    stream = streams[0]
    pix_fmt = (stream.get("pix_fmt") or "").lower()
    transfer = (stream.get("color_transfer") or "").lower()
    return pix_fmt.endswith("10le") or transfer in {"arib-std-b67", "smpte2084"}


def _ffmpeg_tonemap(video_path: Path, out_path: Path, crf: int, preset: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg tonemap failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pass Wav2Lip wrapper.")
    parser.add_argument("--video", type=Path, required=True, help="Source video.")
    parser.add_argument("--audio", type=Path, required=True, help="Audio wav (16k or 24k).")
    parser.add_argument("--output", type=Path, required=True, help="Output mp4 path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Wav2Lip checkpoint.")
    parser.add_argument("--wav2lip_dir", type=Path, default=Path("lib/Wav2Lip"))
    parser.add_argument("--resize_factor", type=int, default=1)
    parser.add_argument("--pads", type=str, default="0 10 0 0")
    parser.add_argument("--face_det_batch_size", type=int, default=16)
    parser.add_argument("--wav2lip_batch_size", type=int, default=128)
    parser.add_argument("--crop", type=str, default="0 -1 0 -1")
    parser.add_argument("--box", type=str, default="-1 -1 -1 -1")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--nosmooth", action="store_true")
    parser.add_argument("--fourcc", type=str, default="DIVX")
    parser.add_argument("--wav2lip_crf", type=int, default=18)
    parser.add_argument("--wav2lip_preset", type=str, default="slow")
    parser.add_argument("--no_tonemap", action="store_true")
    parser.add_argument("--tonemap_crf", type=int, default=15)
    parser.add_argument("--tonemap_preset", type=str, default="slow")
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--save_cache", action="store_true")

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    wav2lip_dir = args.wav2lip_dir.resolve()
    inference_py = wav2lip_dir / "inference.py"
    if not inference_py.exists():
        raise FileNotFoundError(f"Wav2Lip inference not found: {inference_py}")

    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir: Path | None = None
    input_video = args.video
    if not args.no_tonemap and _needs_tonemap(args.video):
        if args.save_cache and args.cache_dir:
            tmp_dir = args.cache_dir / f"{args.video.stem}_tonemap"
            tmp_dir.mkdir(parents=True, exist_ok=True)
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="lipsync_lib_"))
        sdr_path = tmp_dir / f"{args.video.stem}_sdr.mp4"
        print(f"[prep] Tonemapping HDR -> SDR -> {sdr_path}")
        _ffmpeg_tonemap(args.video, sdr_path, args.tonemap_crf, args.tonemap_preset)
        input_video = sdr_path

    env = os.environ.copy()
    env["WAV2LIP_FOURCC"] = args.fourcc
    env["WAV2LIP_CRF"] = str(args.wav2lip_crf)
    env["WAV2LIP_PRESET"] = args.wav2lip_preset

    cmd = [
        sys.executable,
        str(inference_py),
        "--checkpoint_path",
        str(args.checkpoint),
        "--face",
        str(input_video),
        "--audio",
        str(args.audio),
        "--outfile",
        str(args.output),
        "--resize_factor",
        str(args.resize_factor),
        "--pads",
        *args.pads.split(),
        "--face_det_batch_size",
        str(args.face_det_batch_size),
        "--wav2lip_batch_size",
        str(args.wav2lip_batch_size),
        "--crop",
        *args.crop.split(),
        "--box",
        *args.box.split(),
    ]
    if args.rotate:
        cmd.append("--rotate")
    if args.nosmooth:
        cmd.append("--nosmooth")

    print("[run] Starting Wav2Lip inference...")
    _run(cmd, cwd=wav2lip_dir, env=env)
    print(f"[done] Saved: {args.output}")

    if tmp_dir and not args.save_cache:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
