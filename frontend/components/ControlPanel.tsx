import React, { useCallback, useRef, useState } from 'react';
import { useSpeechToText } from '../hooks/useSpeechToText';
import { useMobileVoiceInput } from '../mobile/useMobileVoiceInput';

type Mode = 'chat' | 'tts';

type ControlPanelProps = {
  onSendChat: (text: string) => Promise<void> | void;
  onSendDirect: (text: string) => Promise<void> | void;
  onInterrupt?: () => Promise<void> | void;
  disabled?: boolean;
  variant?: 'card' | 'embedded';
  apiBase?: string;
};

const ControlPanel: React.FC<ControlPanelProps> = ({
  onSendChat,
  onSendDirect,
  onInterrupt,
  disabled,
  variant = 'card',
  apiBase = '',
}) => {
  const [mode, setMode] = useState<Mode>('tts');
  const [text, setText] = useState('');
  const [chatText, setChatText] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isDisabled = !!disabled;
  const isSecureLikeContext =
    typeof window !== 'undefined' &&
    (window.isSecureContext || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

  const handleSend = useCallback(
    async (selectedMode: Mode, inputText: string) => {
      if (!inputText.trim()) return;
      if (onInterrupt) {
        await onInterrupt();
      }
      setIsSending(true);
      try {
        if (selectedMode === 'chat') {
          await onSendChat(inputText);
        } else {
          await onSendDirect(inputText);
        }
      } finally {
        setIsSending(false);
      }
    },
    [onInterrupt, onSendChat, onSendDirect],
  );

  const onSpeechResult = useCallback(
    async (spoken: string) => {
      setChatText(spoken);
      await handleSend('chat', spoken);
    },
    [handleSend],
  );

  const { isListening, startListening, stopListening, hasSupport, transcript } = useSpeechToText(onSpeechResult);
  const {
    hasRecorderSupport,
    isRecording: isMobileRecording,
    isTranscribing: isMobileTranscribing,
    error: mobileVoiceError,
    startRecording: startMobileRecording,
    stopRecording: stopMobileRecording,
    transcribeFile,
    clearError: clearMobileVoiceError,
  } = useMobileVoiceInput({
    apiBase,
    onFinalText: onSpeechResult,
  });
  const canUseNativeSpeech = hasSupport && isSecureLikeContext;
  const canUseMobileRecorder = !canUseNativeSpeech && hasRecorderSupport && isSecureLikeContext && Boolean(apiBase);
  const canUseCaptureFallback = !canUseNativeSpeech && Boolean(apiBase);
  const canUseAnyVoiceInput = canUseNativeSpeech || canUseMobileRecorder || canUseCaptureFallback;
  const isVoiceListening = canUseNativeSpeech ? isListening : isMobileRecording;
  const voiceButtonLabel = isMobileTranscribing
    ? 'Transcribing…'
    : isVoiceListening
      ? 'Listening…'
      : 'Voice Input';

  const handleVoiceInput = useCallback(async () => {
    if (isDisabled || mode !== 'chat') return;
    setCaptureError(null);
    clearMobileVoiceError();
    if (onInterrupt) {
      await onInterrupt();
    }

    if (canUseNativeSpeech) {
      if (isListening) {
        stopListening();
      } else {
        startListening();
      }
      return;
    }

    if (canUseMobileRecorder) {
      if (isMobileRecording) {
        stopMobileRecording();
        return;
      }
      await startMobileRecording();
      return;
    }

    if (canUseCaptureFallback) {
      fileInputRef.current?.click();
      return;
    }

    setCaptureError('Voice input is unavailable in this browser/context.');
  }, [
    canUseCaptureFallback,
    canUseMobileRecorder,
    canUseNativeSpeech,
    clearMobileVoiceError,
    isDisabled,
    isListening,
    isMobileRecording,
    mode,
    onInterrupt,
    startListening,
    stopListening,
    startMobileRecording,
    stopMobileRecording,
  ]);

  const containerClass =
    variant === 'embedded'
      ? 'w-full'
      : 'w-full rounded-2xl border border-slate-800 bg-slate-950/90 p-4 backdrop-blur';

  const inputClass =
    variant === 'embedded'
      ? 'h-24 w-full rounded-xl border border-slate-800 bg-slate-950 p-3 text-sm text-white outline-none'
      : 'h-24 w-full rounded-xl border border-slate-800 bg-slate-900/70 p-3 text-sm text-white outline-none';

  return (
    <div className={containerClass}>
      <div className="mb-3 flex gap-2">
        <button
          onClick={() => setMode('chat')}
          className={`flex-1 rounded-lg px-3 py-2 text-xs font-bold transition-all ${
            mode === 'chat' ? 'bg-teal-600 text-white' : 'bg-slate-900 text-slate-400'
          }`}
        >
          Chat (LLM)
        </button>
        <button
          onClick={() => setMode('tts')}
          className={`flex-1 rounded-lg px-3 py-2 text-xs font-bold transition-all ${
            mode === 'tts' ? 'bg-teal-600 text-white' : 'bg-slate-900 text-slate-400'
          }`}
        >
          Say (TTS)
        </button>
      </div>

      <textarea
        value={mode === 'chat' ? (transcript || chatText) : text}
        onChange={(e) => (mode === 'chat' ? setChatText(e.target.value) : setText(e.target.value))}
        placeholder={mode === 'chat' ? 'Ask the assistant…' : 'Type what to say…'}
        className={inputClass}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend(mode, mode === 'chat' ? chatText : text);
          }
        }}
      />

      <div className="mt-3 flex items-center justify-between text-xs text-slate-400 gap-2">
        <span>{mode === 'chat' ? chatText.length : text.length} chars</span>
        <div className="flex items-center gap-2">
          {mode === 'chat' && (
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              capture="user"
              className="hidden"
              onChange={async (e) => {
                const file = e.target.files?.[0] || null;
                if (!file) {
                  e.currentTarget.value = '';
                  return;
                }
                setCaptureError(null);
                clearMobileVoiceError();
                await transcribeFile(file);
                e.currentTarget.value = '';
              }}
            />
          )}
          {mode === 'chat' && (
            <button
              onClick={handleVoiceInput}
              disabled={isDisabled || isMobileTranscribing || !canUseAnyVoiceInput}
              className={`rounded-lg px-3 py-2 text-xs font-bold ${
                isVoiceListening ? 'bg-rose-500 text-white' : 'bg-slate-800 text-slate-200'
              }`}
            >
              {voiceButtonLabel}
            </button>
          )}
          <button
            onClick={() => handleSend(mode, mode === 'chat' ? chatText : text)}
            disabled={isDisabled}
            className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-bold text-white"
          >
            {isSending ? 'Processing…' : mode === 'chat' ? 'Send to LLM' : 'Generate'}
          </button>
        </div>
      </div>

      {mode === 'chat' && !canUseNativeSpeech && !isSecureLikeContext && (
        <p className="mt-2 text-[11px] text-slate-500">
          iOS live mic capture needs HTTPS. Voice Input will open the recorder fallback on this connection.
        </p>
      )}
      {mode === 'chat' && !canUseAnyVoiceInput && (
        <p className="mt-2 text-[11px] text-slate-500">
          Voice input is unavailable in this browser/context. Use HTTPS and allow microphone access on iOS.
        </p>
      )}
      {captureError && (
        <p className="mt-2 text-[11px] text-rose-400">{captureError}</p>
      )}
      {mobileVoiceError && (
        <p className="mt-2 text-[11px] text-rose-400">{mobileVoiceError}</p>
      )}
    </div>
  );
};

export default ControlPanel;
