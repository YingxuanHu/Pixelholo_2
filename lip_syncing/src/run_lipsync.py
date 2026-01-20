import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed.")


def _run_live(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Command failed.")


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
    parser = argparse.ArgumentParser(description="One-shot lipsync wrapper.")
    parser.add_argument("--video", type=Path, required=True, help="Source video with audio.")
    parser.add_argument("--output", type=Path, default=None, help="Final mp4 output path.")
    parser.add_argument("--keep_audio", action="store_true", help="Keep extracted audio file.")
    parser.add_argument("--audio_out", type=Path, default=None, help="Optional audio output path.")

    parser.add_argument("--chunk_sec", type=float, default=1.0)
    parser.add_argument("--loop_sec", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--resize_factor", type=int, default=2)
    parser.add_argument("--loop_crf", type=int, default=18)
    parser.add_argument("--loop_preset", type=str, default="slow")
    parser.add_argument("--chunk_crf", type=int, default=18)
    parser.add_argument("--chunk_preset", type=str, default="slow")
    parser.add_argument("--face_det_batch_size", type=int, default=16)
    parser.add_argument("--wav2lip_batch_size", type=int, default=128)
    parser.add_argument("--pads", type=str, default="0 10 0 0")
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--nosmooth", action="store_true")
    parser.add_argument("--wav2lip_crf", type=int, default=18)
    parser.add_argument("--wav2lip_preset", type=str, default="slow")
    parser.add_argument("--concat_mode", choices=("copy", "reencode"), default="copy")
    parser.add_argument("--concat_crf", type=int, default=18)
    parser.add_argument("--concat_preset", type=str, default="slow")
    parser.add_argument("--no_tonemap", action="store_true")
    parser.add_argument("--tonemap_crf", type=int, default=15)
    parser.add_argument("--tonemap_preset", type=str, default="slow")
    parser.add_argument("--wav2lip_dir", type=Path, default=Path("lib/Wav2Lip"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/wav2lip_gan.pth"))

    args = parser.parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    out_path = args.output
    if out_path is None:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.video.stem}_lipsync.mp4"

    tmp_dir = Path(tempfile.mkdtemp(prefix="lipsync_wrap_"))
    if args.audio_out:
        audio_path = args.audio_out
        audio_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        audio_path = tmp_dir / f"{args.video.stem}_audio.wav"

    input_video = args.video
    if not args.no_tonemap and _needs_tonemap(args.video):
        sdr_path = tmp_dir / f"{args.video.stem}_sdr.mp4"
        _ffmpeg_tonemap(args.video, sdr_path, args.tonemap_crf, args.tonemap_preset)
        input_video = sdr_path

    _ffmpeg_extract_audio(input_video, audio_path)

    print("[run] Starting chunked lipsync...")
    chunked = Path(__file__).with_name("chunked_lipsync.py")
    cmd = [
        sys.executable,
        str(chunked),
        "--video",
        str(input_video),
        "--audio",
        str(audio_path),
        "--output",
        str(out_path),
        "--chunk_sec",
        str(args.chunk_sec),
        "--loop_sec",
        str(args.loop_sec),
        "--fps",
        str(args.fps),
        "--resize_factor",
        str(args.resize_factor),
        "--loop_crf",
        str(args.loop_crf),
        "--loop_preset",
        args.loop_preset,
        "--chunk_crf",
        str(args.chunk_crf),
        "--chunk_preset",
        args.chunk_preset,
        "--face_det_batch_size",
        str(args.face_det_batch_size),
        "--wav2lip_batch_size",
        str(args.wav2lip_batch_size),
        "--pads",
        args.pads,
        "--fourcc",
        args.fourcc,
        "--wav2lip_crf",
        str(args.wav2lip_crf),
        "--wav2lip_preset",
        args.wav2lip_preset,
        "--loop_crf",
        str(args.loop_crf),
        "--loop_preset",
        args.loop_preset,
        "--chunk_crf",
        str(args.chunk_crf),
        "--chunk_preset",
        args.chunk_preset,
        "--concat_mode",
        args.concat_mode,
        "--concat_crf",
        str(args.concat_crf),
        "--concat_preset",
        args.concat_preset,
        "--wav2lip_dir",
        str(args.wav2lip_dir),
        "--checkpoint",
        str(args.checkpoint),
    ]
    if args.nosmooth:
        cmd.append("--nosmooth")

    _run_live(cmd)
    print(f"[done] Saved: {out_path}")

    if not args.keep_audio and not args.audio_out:
        print(f"[cleanup] Removing temp dir {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
