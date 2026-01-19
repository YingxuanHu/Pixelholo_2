import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    run_cwd = str(cwd) if cwd is not None else None
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed.")


def _run_live(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    run_cwd = str(cwd) if cwd is not None else None
    result = subprocess.run(cmd, cwd=run_cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError("Command failed.")


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return float(result.stdout.strip())


def _ffprobe_fps(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    raw = result.stdout.strip()
    if "/" in raw:
        num, den = raw.split("/", 1)
        return float(num) / float(den)
    return float(raw)


def _make_loop(
    video_path: Path,
    loop_path: Path,
    loop_sec: float,
    fps: float,
    crf: int,
    preset: str,
) -> None:
    print(f"[prep] Building loop video ({loop_sec:.2f}s @ {fps:.2f}fps, crf={crf})...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-t",
        f"{loop_sec:.3f}",
        "-r",
        f"{fps:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(loop_path),
    ]
    _run(cmd)


def _make_chunk_audio(audio_path: Path, out_path: Path, start: float, dur: float) -> None:
    print(f"[prep] Audio chunk {out_path.name} ({start:.2f}s -> {start + dur:.2f}s)")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(audio_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]
    _run(cmd)


def _make_chunk_video(
    loop_path: Path,
    out_path: Path,
    dur: float,
    fps: float,
    crf: int,
    preset: str,
) -> None:
    print(f"[prep] Video chunk {out_path.name} ({dur:.2f}s, crf={crf})")
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(loop_path),
        "-t",
        f"{dur:.3f}",
        "-r",
        f"{fps:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(out_path),
    ]
    _run(cmd)


def _run_wav2lip(
    wav2lip_dir: Path,
    checkpoint: Path,
    face: Path,
    audio: Path,
    out: Path,
    resize_factor: int,
    face_det_batch_size: int,
    wav2lip_batch_size: int,
    pads: str,
    nosmooth: bool,
    fourcc: str,
    wav2lip_crf: int,
    wav2lip_preset: str,
) -> None:
    wav2lip_dir = wav2lip_dir.resolve()
    checkpoint = checkpoint.resolve()
    (wav2lip_dir / "temp").mkdir(parents=True, exist_ok=True)
    inference_path = wav2lip_dir / "inference.py"
    cmd = [
        sys.executable,
        str(inference_path),
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(face),
        "--audio",
        str(audio),
        "--outfile",
        str(out),
        "--resize_factor",
        str(resize_factor),
        "--face_det_batch_size",
        str(face_det_batch_size),
        "--wav2lip_batch_size",
        str(wav2lip_batch_size),
        "--pads",
        *pads.split(),
    ]
    if nosmooth:
        cmd.append("--nosmooth")
    env = dict(**os.environ)
    env["WAV2LIP_FOURCC"] = fourcc
    env["WAV2LIP_CRF"] = str(wav2lip_crf)
    env["WAV2LIP_PRESET"] = wav2lip_preset
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[wav2lip] Running on {face.name} + {audio.name}")
    _run_live(cmd, cwd=wav2lip_dir, env=env)


def _concat_videos(files: List[Path], output: Path, mode: str, crf: int, preset: str) -> None:
    if len(files) == 1:
        print(f"[concat] Single chunk, copying to {output}")
        shutil.copy2(files[0], output)
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        for path in files:
            handle.write(f"file '{path.as_posix()}'\n")
        list_path = Path(handle.name)
    try:
        if mode == "copy":
            print("[concat] Attempting stream copy (no re-encode).")
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ]
            try:
                _run(cmd)
                return
            except RuntimeError:
                mode = "reencode"
        print(f"[concat] Re-encoding with CRF={crf}, preset={preset}.")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
        _run(cmd)
    finally:
        list_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked Wav2Lip runner.")
    parser.add_argument("--video", type=Path, required=True, help="Source portrait video.")
    parser.add_argument("--audio", type=Path, required=True, help="Target audio (wav/mp3).")
    parser.add_argument("--output", type=Path, required=True, help="Output mp4.")
    parser.add_argument("--wav2lip_dir", type=Path, default=Path("lib/Wav2Lip"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/wav2lip_gan.pth"))
    parser.add_argument("--fps", type=float, default=0.0, help="Override FPS (default: detect).")
    parser.add_argument("--chunk_sec", type=float, default=1.0, help="Chunk size in seconds.")
    parser.add_argument("--loop_sec", type=float, default=10.0, help="Loop video length in seconds.")
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=2,
        help="Wav2Lip resize factor (reduces face detection load).",
    )
    parser.add_argument(
        "--loop_crf",
        type=int,
        default=18,
        help="CRF for loop video encoding (lower is higher quality).",
    )
    parser.add_argument(
        "--loop_preset",
        type=str,
        default="slow",
        help="x264 preset for loop encoding.",
    )
    parser.add_argument(
        "--chunk_crf",
        type=int,
        default=18,
        help="CRF for chunk video encoding (lower is higher quality).",
    )
    parser.add_argument(
        "--chunk_preset",
        type=str,
        default="slow",
        help="x264 preset for chunk encoding.",
    )
    parser.add_argument(
        "--face_det_batch_size",
        type=int,
        default=16,
        help="Wav2Lip face detector batch size.",
    )
    parser.add_argument(
        "--wav2lip_batch_size",
        type=int,
        default=128,
        help="Wav2Lip model batch size.",
    )
    parser.add_argument(
        "--pads",
        type=str,
        default="0 10 0 0",
        help="Padding top bottom left right (space-separated).",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="Intermediate video codec (e.g. MJPG or DIVX).",
    )
    parser.add_argument(
        "--wav2lip_crf",
        type=int,
        default=18,
        help="CRF for Wav2Lip output encode (lower is higher quality).",
    )
    parser.add_argument(
        "--wav2lip_preset",
        type=str,
        default="slow",
        help="x264 preset for Wav2Lip output encode.",
    )
    parser.add_argument(
        "--nosmooth",
        action="store_true",
        help="Disable face-box smoothing (can preserve sharp edges).",
    )
    parser.add_argument(
        "--concat_mode",
        choices=("copy", "reencode"),
        default="copy",
        help="Concatenation mode (copy preserves quality).",
    )
    parser.add_argument(
        "--concat_crf",
        type=int,
        default=18,
        help="CRF for concat re-encode (lower is higher quality).",
    )
    parser.add_argument(
        "--concat_preset",
        type=str,
        default="slow",
        help="x264 preset for concat re-encode.",
    )
    parser.add_argument(
        "--min_chunk_sec",
        type=float,
        default=0.25,
        help="Skip trailing chunks shorter than this duration.",
    )
    parser.add_argument("--keep_temp", action="store_true", help="Keep temp chunk files.")

    args = parser.parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio}")
    if not (args.wav2lip_dir / "inference.py").exists():
        raise FileNotFoundError(f"Wav2Lip not found at {args.wav2lip_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    fps = args.fps if args.fps > 0 else _ffprobe_fps(args.video)
    audio_dur = _ffprobe_duration(args.audio)
    total_chunks = max(1, math.ceil(audio_dur / args.chunk_sec))

    print(f"[info] Video: {args.video}")
    print(f"[info] Audio: {args.audio} ({audio_dur:.2f}s)")
    print(f"[info] FPS: {fps:.2f} | Chunks: {total_chunks} | Chunk size: {args.chunk_sec:.2f}s")
    print(f"[info] Resize factor: {args.resize_factor} | Pads: {args.pads}")

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="lipsync_"))

    loop_path = tmp_dir / "loop.mp4"
    _make_loop(args.video, loop_path, args.loop_sec, fps, args.loop_crf, args.loop_preset)

    chunk_outputs: list[Path] = []
    for idx in range(total_chunks):
        start = idx * args.chunk_sec
        remaining = audio_dur - start
        if remaining <= 0:
            break
        dur = min(args.chunk_sec, remaining)
        if dur < args.min_chunk_sec:
            print(f"[skip] Trailing chunk {idx + 1} too short ({dur:.3f}s).")
            break
        audio_chunk = tmp_dir / f"audio_{idx:04d}.wav"
        video_chunk = tmp_dir / f"video_{idx:04d}.mp4"
        out_chunk = tmp_dir / f"out_{idx:04d}.mp4"
        _make_chunk_audio(args.audio, audio_chunk, start, dur)
        _make_chunk_video(loop_path, video_chunk, dur, fps, args.chunk_crf, args.chunk_preset)
        _run_wav2lip(
            args.wav2lip_dir,
            args.checkpoint,
            video_chunk,
            audio_chunk,
            out_chunk,
            args.resize_factor,
            args.face_det_batch_size,
            args.wav2lip_batch_size,
            args.pads,
            args.nosmooth,
            args.fourcc,
            args.wav2lip_crf,
            args.wav2lip_preset,
        )
        chunk_outputs.append(out_chunk)
        print(f"Chunk {idx + 1}/{total_chunks} done.")

    _concat_videos(chunk_outputs, args.output, args.concat_mode, args.concat_crf, args.concat_preset)
    print(f"Saved: {args.output}")

    if not args.keep_temp:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
