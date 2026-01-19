"""
Environment-level monkey patches loaded automatically before user code.

We specifically shim torchaudio's removed list_audio_backends API so that
packages like SpeechBrain (pulled in by MFA) keep working with newer
nightly torchaudio wheels.
"""
try:
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends() -> list[str]:
            # Nightly wheels no longer expose backend enumeration; MFA just
            # needs *something* to satisfy SpeechBrain's sanity check.
            return ["sox_io"]

        torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]
except Exception:
    pass

