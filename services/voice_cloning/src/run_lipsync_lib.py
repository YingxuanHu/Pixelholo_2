import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed.")


def _ffmpeg_extract_audio(video_path: Path, audio_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH.")
    print(f"[prep] Extracting audio -> {audio_path}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    _run(cmd)


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
    print(f"[prep] Tonemapping HDR -> SDR -> {out_path}")
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
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PixelHolo-style LipSync runner (single-pass, cached)."
    )
    parser.add_argument("--video", type=Path, required=True, help="Source video.")
    parser.add_argument("--audio", type=Path, default=None, help="Optional 16k mono wav.")
    parser.add_argument("--output", type=Path, required=True, help="Output mp4 path.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/wav2lip_gan.pth"))
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--cache_dir", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--nosmooth", action="store_true")
    parser.add_argument("--save_cache", action="store_true")
    parser.add_argument("--no_tonemap", action="store_true")
    parser.add_argument("--tonemap_crf", type=int, default=15)
    parser.add_argument("--tonemap_preset", type=str, default="slow")
    parser.add_argument("--keep_audio", action="store_true")

    args = parser.parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    try:
        from lipsync import LipSync
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: lipsync. Install it with `pip install lipsync==0.1.0`."
        ) from exc

    tmp_dir = Path(tempfile.mkdtemp(prefix="lipsync_lib_"))
    input_video = args.video
    if not args.no_tonemap and _needs_tonemap(args.video):
        sdr_path = tmp_dir / f"{args.video.stem}_sdr.mp4"
        _ffmpeg_tonemap(args.video, sdr_path, args.tonemap_crf, args.tonemap_preset)
        input_video = sdr_path

    if args.audio is None:
        audio_path = tmp_dir / f"{args.video.stem}_audio.wav"
        _ffmpeg_extract_audio(input_video, audio_path)
    else:
        audio_path = args.audio

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("[run] Loading LipSync model...")
    lip = LipSync(
        model="wav2lip",
        checkpoint_path=str(args.checkpoint),
        nosmooth=args.nosmooth,
        device=args.device,
        cache_dir=str(args.cache_dir),
        img_size=args.img_size,
        save_cache=args.save_cache,
    )
    print("[run] Running lip-sync...")
    lip.sync(str(input_video), str(audio_path), str(args.output))
    print(f"[done] Saved: {args.output}")

    if args.audio is None and not args.keep_audio:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
