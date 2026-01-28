from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf


@dataclass
class StyleBank:
    neutral: list[Path]
    excited: list[Path]
    serious: list[Path]
    inquisitive: list[Path]


_STYLE_CACHE: dict[str, StyleBank] = {}


def _score_clip(path: Path) -> tuple[float, float]:
    """Return (rms, f0_median) for a clip. Lower f0 = deeper tone."""
    try:
        import librosa
    except Exception:
        return 0.0, 0.0

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio, _ = librosa.effects.trim(audio, top_db=35)
    if audio.size < sr // 2:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    try:
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0_median = float(np.median(f0)) if f0.size else 0.0
    except Exception:
        f0_median = 0.0
    return rms, f0_median


def _build_style_bank(profile_dir: Path) -> StyleBank:
    wav_dir = profile_dir / "processed_wavs"
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        return StyleBank([], [], [], [])

    scored = []
    for path in wavs:
        rms, f0 = _score_clip(path)
        scored.append((path, rms, f0))

    # Sort by energy then by f0
    scored.sort(key=lambda x: (x[1], x[2]))
    n = len(scored)
    if n < 5:
        # Not enough clips: just treat all as neutral
        return StyleBank([p for p, *_ in scored], [], [], [])

    # Partition into quantiles
    low = scored[: max(1, n // 5)]
    high = scored[-max(1, n // 5) :]
    mid = scored[n // 5 : -n // 5] if n // 5 < n - n // 5 else scored

    neutral = [p for p, *_ in mid] or [p for p, *_ in scored]
    excited = [p for p, *_ in high] or neutral
    serious = [p for p, *_ in low] or neutral

    # Inquisitive: higher pitch but not necessarily loud
    by_f0 = sorted(scored, key=lambda x: x[2])
    inquisitive = [p for p, *_ in by_f0[-max(1, n // 5) :]] or neutral

    return StyleBank(neutral, excited, serious, inquisitive)


def get_style_bank(profile_dir: Path) -> StyleBank:
    key = str(profile_dir.resolve())
    if key not in _STYLE_CACHE:
        _STYLE_CACHE[key] = _build_style_bank(profile_dir)
    return _STYLE_CACHE[key]


def pick_style_ref(text: str, profile_dir: Path, fallback: Path) -> Path:
    bank = get_style_bank(profile_dir)
    text_l = text.lower()
    if "!" in text or any(w in text_l for w in ("wow", "awesome", "amazing", "incredible")):
        if bank.excited:
            return np.random.choice(bank.excited)
    if "?" in text:
        if bank.inquisitive:
            return np.random.choice(bank.inquisitive)
    if any(w in text_l for w in ("warning", "important", "serious", "critical")):
        if bank.serious:
            return np.random.choice(bank.serious)
    if bank.neutral:
        return np.random.choice(bank.neutral)
    return fallback
