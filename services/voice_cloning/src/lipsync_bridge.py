import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from config import LIP_SYNCING_DIR, avatar_cache_dir


class LipSyncBridge:
    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str | None = None,
        img_size: int = 96,
        wav2lip_batch_size: int = 128,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.frame_idx = 0
        self.frames: np.ndarray | None = None
        self.coords: np.ndarray | None = None
        self.fps: float = 25.0

        lip_dir = LIP_SYNCING_DIR / "lib" / "Wav2Lip"
        if not lip_dir.exists():
            raise FileNotFoundError(f"Wav2Lip repo not found at {lip_dir}")
        sys.path.insert(0, str(lip_dir))
        from models import Wav2Lip  # type: ignore
        import audio as wav2lip_audio  # type: ignore

        self._wav_audio = wav2lip_audio
        self.model = Wav2Lip().to(self.device)

        ckpt = checkpoint_path or (LIP_SYNCING_DIR / "models" / "wav2lip_gan.pth")
        if not ckpt.exists():
            raise FileNotFoundError(f"Wav2Lip checkpoint not found: {ckpt}")
        checkpoint = torch.load(ckpt, map_location=self.device)
        state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def load_profile(self, profile: str) -> None:
        cache_dir = avatar_cache_dir(profile)
        frames_path = cache_dir / "frames.npy"
        coords_path = cache_dir / "coords.npy"
        meta_path = cache_dir / "meta.json"
        if not frames_path.exists() or not coords_path.exists():
            raise FileNotFoundError(
                f"Avatar cache missing for {profile}. Run preprocess with avatar baking."
            )
        self.frames = np.load(frames_path)
        self.coords = np.load(coords_path)
        if meta_path.exists():
            meta = meta_path.read_text()
            try:
                import json

                data = json.loads(meta)
                self.fps = float(data.get("fps", self.fps))
            except Exception:
                pass
        self.frame_idx = 0

    def _mel_chunks(self, audio_16k: np.ndarray) -> list[np.ndarray]:
        mel = self._wav_audio.melspectrogram(audio_16k)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel contains NaN.")
        mel_step_size = 16
        mel_chunks = []
        mel_idx_multiplier = 80.0 / float(self.fps or 25.0)
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - mel_step_size :])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        return mel_chunks

    def _batch_iter(
        self,
        frames: list[np.ndarray],
        coords: list[list[int]],
        mels: list[np.ndarray],
    ) -> Iterable[tuple[np.ndarray, np.ndarray, list[np.ndarray], list[list[int]]]]:
        img_batch, mel_batch, frame_batch, coord_batch = [], [], [], []
        for frame, coord, mel in zip(frames, coords, mels):
            y1, y2, x1, x2 = coord
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (self.img_size, self.img_size))
            img_batch.append(face)
            mel_batch.append(mel)
            frame_batch.append(frame)
            coord_batch.append(coord)
            if len(img_batch) >= self.wav2lip_batch_size:
                yield self._prepare_batch(img_batch, mel_batch, frame_batch, coord_batch)
                img_batch, mel_batch, frame_batch, coord_batch = [], [], [], []
        if img_batch:
            yield self._prepare_batch(img_batch, mel_batch, frame_batch, coord_batch)

    def _prepare_batch(self, img_batch, mel_batch, frame_batch, coord_batch):
        img_batch = np.asarray(img_batch)
        mel_batch = np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, self.img_size // 2 :] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        return img_batch, mel_batch, frame_batch, coord_batch

    def sync_chunk(self, audio_16k: np.ndarray) -> list[np.ndarray]:
        if self.frames is None or self.coords is None:
            raise RuntimeError("Avatar cache not loaded.")
        if audio_16k.size == 0:
            return []
        mel_chunks = self._mel_chunks(audio_16k)
        if not mel_chunks:
            return []
        frames = []
        coords = []
        for _ in mel_chunks:
            idx = self.frame_idx % len(self.frames)
            frames.append(self.frames[idx].copy())
            coords.append(self.coords[idx].tolist())
            self.frame_idx += 1

        output_frames: list[np.ndarray] = []
        for img_batch, mel_batch, frame_batch, coord_batch in self._batch_iter(
            frames, coords, mel_chunks
        ):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for p, f, c in zip(pred, frame_batch, coord_batch):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                output_frames.append(f)
        return output_frames
