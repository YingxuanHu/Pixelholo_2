import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    LIP_SYNCING_DIR,
    PROFILE_TYPE_AVATAR,
    PROFILE_TYPE_VOICE,
    avatar_cache_dir,
)


def _smooth_boxes(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    smoothed = boxes.copy().astype(np.float32)
    for i in range(len(smoothed)):
        start = max(0, i - window + 1)
        smoothed[i] = smoothed[start : i + 1].mean(axis=0)
    return smoothed.round().astype(np.int32)


def _apply_loop_crossfade(frames: list[np.ndarray], fade_frames: int) -> list[np.ndarray]:
    if fade_frames <= 0 or fade_frames * 2 >= len(frames):
        return frames
    total = len(frames)
    for i in range(fade_frames):
        alpha = float(i + 1) / float(fade_frames)
        tail_idx = total - fade_frames + i
        head_idx = i
        blended = cv2.addWeighted(frames[tail_idx], 1.0 - alpha, frames[head_idx], alpha, 0)
        frames[tail_idx] = blended
    return frames


def _load_detector(device: str = "cuda"):
    lip_dir = LIP_SYNCING_DIR / "lib" / "Wav2Lip"
    if not lip_dir.exists():
        raise FileNotFoundError(f"Wav2Lip repo not found at {lip_dir}")
    sys.path.insert(0, str(lip_dir))
    import face_detection  # type: ignore

    return face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device
    )


def bake_avatar(
    profile: str,
    video_path: Path,
    profile_type: str = PROFILE_TYPE_AVATAR,
    fps: float = 25.0,
    start_sec: float = 5.0,
    loop_sec: float = 10.0,
    loop_fade_sec: float = 0.0,
    resize_factor: int = 1,
    pads: tuple[int, int, int, int] = (0, 10, 0, 0),
    batch_size: int = 16,
    nosmooth: bool = False,
    blur_background: bool = True,
    blur_kernel: int = 31,
    device: str = "cuda",
) -> Path:
    cache_dir = avatar_cache_dir(profile, profile_type)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    fps = fps or src_fps or 25.0
    frame_interval = max(1, int(round(src_fps / fps))) if src_fps else 1
    start_frame = int(round(start_sec * src_fps)) if src_fps and start_sec > 0 else 0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    max_frames = int(loop_sec * fps) if loop_sec > 0 else None

    frames: list[np.ndarray] = []
    frame_index = start_frame if start_frame > 0 else 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame and frame_index < start_frame:
            frame_index += 1
            continue
        if frame_index % frame_interval != 0:
            frame_index += 1
            continue
        if resize_factor > 1:
            frame = cv2.resize(
                frame,
                (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor),
            )
        frames.append(frame)
        frame_index += 1
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted for avatar baking.")

    if loop_fade_sec and fps:
        fade_frames = int(round(loop_fade_sec * fps))
        frames = _apply_loop_crossfade(frames, fade_frames)

    detector = _load_detector(device)
    preds: list[np.ndarray | None] = []
    for i in range(0, len(frames), batch_size):
        batch = np.array(frames[i : i + batch_size])
        preds.extend(detector.get_detections_for_batch(batch))

    coords: list[list[int]] = []
    last_good: list[int] | None = None
    top_pad, bottom_pad, left_pad, right_pad = pads
    for pred, frame in zip(preds, frames):
        if pred is None:
            if last_good is None:
                raise RuntimeError("Face not detected in the first frame.")
            coords.append(last_good)
            continue
        values = list(pred)
        if len(values) < 4:
            raise RuntimeError(f"Face detector returned invalid box: {pred}")
        x1, y1, x2, y2 = values[:4]
        y1 = max(0, int(y1) - top_pad)
        y2 = min(frame.shape[0], int(y2) + bottom_pad)
        x1 = max(0, int(x1) - left_pad)
        x2 = min(frame.shape[1], int(x2) + right_pad)
        box = [y1, y2, x1, x2]
        coords.append(box)
        last_good = box

    coords_arr = np.array(coords, dtype=np.int32)
    if not nosmooth:
        coords_arr = _smooth_boxes(coords_arr, window=5)

    if blur_background:
        k = max(3, int(blur_kernel) // 2 * 2 + 1)
        blurred_frames: list[np.ndarray] = []
        for frame, box in zip(frames, coords_arr):
            y1, y2, x1, x2 = box
            blurred = cv2.GaussianBlur(frame, (k, k), 0)
            blurred[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            blurred_frames.append(blurred)
        frames = blurred_frames

    np.save(cache_dir / "frames.npy", np.array(frames, dtype=np.uint8))
    np.save(cache_dir / "coords.npy", coords_arr)

    meta = {
        "profile": profile,
        "source_video": str(video_path),
        "fps": float(fps),
        "frame_count": len(frames),
        "start_sec": float(start_sec),
        "resize_factor": resize_factor,
        "pads": list(pads),
        "loop_fade_sec": float(loop_fade_sec),
        "blur_background": bool(blur_background),
        "blur_kernel": int(blur_kernel),
        "width": int(frames[0].shape[1]),
        "height": int(frames[0].shape[0]),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake avatar frames + face coords for lip-sync.")
    parser.add_argument("--profile", required=True, help="Profile name.")
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument(
        "--profile_type",
        choices=[PROFILE_TYPE_VOICE, PROFILE_TYPE_AVATAR],
        default=PROFILE_TYPE_AVATAR,
        help="Profile type to store avatar cache under.",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--loop_sec", type=float, default=10.0)
    parser.add_argument("--loop_fade_sec", type=float, default=0.0)
    parser.add_argument("--resize_factor", type=int, default=1)
    parser.add_argument("--pads", type=str, default="0 10 0 0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nosmooth", action="store_true")
    parser.add_argument("--no_blur_background", action="store_true")
    parser.add_argument("--blur_kernel", type=int, default=31)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    pads = tuple(int(p) for p in args.pads.split())
    if len(pads) != 4:
        raise ValueError("pads must be 4 ints: top bottom left right")

    cache_dir = bake_avatar(
        profile=args.profile,
        video_path=args.video,
        profile_type=args.profile_type,
        fps=args.fps,
        start_sec=args.start_sec,
        loop_sec=args.loop_sec,
        loop_fade_sec=args.loop_fade_sec,
        resize_factor=args.resize_factor,
        pads=pads,
        batch_size=args.batch_size,
        nosmooth=args.nosmooth,
        blur_background=not args.no_blur_background,
        blur_kernel=args.blur_kernel,
        device=args.device,
    )
    print(f"Avatar cached at {cache_dir}")


if __name__ == "__main__":
    main()
