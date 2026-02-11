import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type MobileVoiceInputOptions = {
  apiBase: string;
  onFinalText: (text: string) => Promise<void> | void;
  language?: string;
};

const pickSupportedMimeType = () => {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return '';
  }
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/mp4',
    'audio/webm',
    'audio/ogg;codecs=opus',
  ];
  return candidates.find((value) => MediaRecorder.isTypeSupported(value)) || '';
};

const toErrorMessage = (error: unknown) => {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      return 'Microphone permission was blocked. Allow mic access and retry.';
    }
    if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      return 'No microphone was found on this device.';
    }
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return 'Voice input failed.';
};

export const useMobileVoiceInput = ({ apiBase, onFinalText, language = 'en' }: MobileVoiceInputOptions) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const monitorTimerRef = useRef<number | null>(null);
  const maxTimerRef = useRef<number | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const analyserDataRef = useRef<Uint8Array | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const startedAtRef = useRef<number>(0);
  const lastVoiceAtRef = useRef<number>(0);
  const mimeTypeRef = useRef<string>('');

  const hasRecorderSupport = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return Boolean(window.MediaRecorder && navigator.mediaDevices?.getUserMedia);
  }, []);

  const cleanupAnalyser = useCallback(() => {
    if (monitorTimerRef.current !== null) {
      window.clearInterval(monitorTimerRef.current);
      monitorTimerRef.current = null;
    }
    if (maxTimerRef.current !== null) {
      window.clearTimeout(maxTimerRef.current);
      maxTimerRef.current = null;
    }
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {
        // ignore cleanup errors
      }
      sourceRef.current = null;
    }
    analyserRef.current = null;
    analyserDataRef.current = null;
    if (audioCtxRef.current) {
      void audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  }, []);

  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
  }, []);

  const cleanupAll = useCallback(() => {
    cleanupAnalyser();
    cleanupStream();
    recorderRef.current = null;
  }, [cleanupAnalyser, cleanupStream]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    if (recorder.state === 'recording') {
      recorder.stop();
    } else {
      cleanupAll();
      setIsRecording(false);
    }
  }, [cleanupAll]);

  const transcribeBlob = useCallback(async (blob: Blob) => {
    const extension = blob.type.includes('mp4') ? 'm4a' : blob.type.includes('ogg') ? 'ogg' : 'webm';
    const file = new File([blob], `voice-input.${extension}`, {
      type: blob.type || 'audio/webm',
    });
    const form = new FormData();
    form.append('file', file);
    form.append('language', language);

    const res = await fetch(`${apiBase}/transcribe_audio`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      throw new Error(await res.text());
    }

    const data = await res.json();
    const spoken = String(data.text || '').trim();
    if (!spoken) {
      throw new Error('No speech detected.');
    }
    await onFinalText(spoken);
  }, [apiBase, language, onFinalText]);

  const transcribeFile = useCallback(async (file: File | null) => {
    if (!file) return;
    if (!apiBase) {
      setError('API base URL is missing.');
      return;
    }
    setError(null);
    setIsTranscribing(true);
    try {
      await transcribeBlob(file);
    } catch (err) {
      setError(toErrorMessage(err));
    } finally {
      setIsTranscribing(false);
    }
  }, [apiBase, transcribeBlob]);

  const startRecording = useCallback(async () => {
    if (isRecording || isTranscribing) return;
    if (!apiBase) {
      setError('API base URL is missing.');
      return;
    }
    if (!hasRecorderSupport) {
      setError('Live microphone capture is unavailable on this iOS browser. Using recorder fallback.');
      return;
    }
    if (!window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      setError('Live microphone capture on iOS requires HTTPS.');
      return;
    }

    setError(null);
    chunksRef.current = [];
    mimeTypeRef.current = pickSupportedMimeType();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const recorder = mimeTypeRef.current
        ? new MediaRecorder(stream, { mimeType: mimeTypeRef.current })
        : new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onerror = () => {
        setError('Microphone recording failed.');
      };
      recorder.onstop = async () => {
        setIsRecording(false);
        cleanupAnalyser();
        cleanupStream();
        recorderRef.current = null;
        const blob = new Blob(chunksRef.current, {
          type: mimeTypeRef.current || 'audio/webm',
        });
        chunksRef.current = [];
        if (blob.size === 0) {
          return;
        }

        setIsTranscribing(true);
        try {
          await transcribeBlob(blob);
        } catch (err) {
          setError(toErrorMessage(err));
        } finally {
          setIsTranscribing(false);
        }
      };

      const Ctx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new Ctx();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyserRef.current = analyser;
      analyserDataRef.current = new Uint8Array(analyser.fftSize);
      startedAtRef.current = performance.now();
      lastVoiceAtRef.current = startedAtRef.current;

      monitorTimerRef.current = window.setInterval(() => {
        const data = analyserDataRef.current;
        const currentAnalyser = analyserRef.current;
        const currentRecorder = recorderRef.current;
        if (!data || !currentAnalyser || !currentRecorder || currentRecorder.state !== 'recording') {
          return;
        }
        currentAnalyser.getByteTimeDomainData(data);
        let sumSquares = 0;
        for (let i = 0; i < data.length; i += 1) {
          const sample = (data[i] - 128) / 128;
          sumSquares += sample * sample;
        }
        const rms = Math.sqrt(sumSquares / data.length);
        const now = performance.now();
        if (rms > 0.03) {
          lastVoiceAtRef.current = now;
        }

        const elapsed = now - startedAtRef.current;
        const silenceFor = now - lastVoiceAtRef.current;
        if (elapsed > 1200 && silenceFor > 900) {
          stopRecording();
        }
      }, 120);

      maxTimerRef.current = window.setTimeout(() => {
        stopRecording();
      }, 20000);

      recorder.start(250);
      setIsRecording(true);
    } catch (err) {
      cleanupAll();
      setIsRecording(false);
      setError(toErrorMessage(err));
    }
  }, [apiBase, cleanupAll, cleanupAnalyser, cleanupStream, hasRecorderSupport, isRecording, isTranscribing, stopRecording, transcribeBlob]);

  const clearError = useCallback(() => setError(null), []);

  useEffect(() => {
    return () => {
      cleanupAll();
    };
  }, [cleanupAll]);

  return {
    hasRecorderSupport,
    isRecording,
    isTranscribing,
    error,
    startRecording,
    stopRecording,
    transcribeFile,
    clearError,
  };
};
