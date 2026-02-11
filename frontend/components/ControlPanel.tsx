import React, { useCallback, useState } from 'react';
import { useSpeechToText } from '../hooks/useSpeechToText';
import { useMobileVoiceInput } from '../mobile/useMobileVoiceInput';

type Mode = 'chat' | 'tts';

type ControlPanelProps = {
  onSendChat: (text: string) => Promise<void> | void;
  onSendDirect: (text: string) => Promise<void> | void;
  onInterrupt?: () => Promise<void> | void;
  onVoiceInputStart?: () => Promise<void> | void;
  disabled?: boolean;
  variant?: 'card' | 'embedded';
  apiBase?: string;
};

const ControlPanel: React.FC<ControlPanelProps> = ({
  onSendChat,
  onSendDirect,
  onInterrupt,
  onVoiceInputStart,
  disabled,
  variant = 'card',
  apiBase = '',
}) => {
  const [mode, setMode] = useState<Mode>('tts');
  const [text, setText] = useState('');
  const [chatText, setChatText] = useState('');
  const [isSending, setIsSending] = useState(false);
  const isDisabled = !!disabled;

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
    hasSupport: hasMobileVoiceSupport,
    isRecording: isMobileRecording,
    isTranscribing: isMobileTranscribing,
    error: mobileVoiceError,
    startRecording: startMobileRecording,
    stopRecording: stopMobileRecording,
    clearError: clearMobileVoiceError,
  } = useMobileVoiceInput({
    apiBase,
    onFinalText: onSpeechResult,
  });
  const canUseAnyVoice = hasSupport || hasMobileVoiceSupport;
  const isVoiceListening = hasSupport ? isListening : isMobileRecording;
  const startListeningSafe = useCallback(async () => {
    if (onInterrupt) {
      await onInterrupt();
    }
    if (onVoiceInputStart) {
      await onVoiceInputStart();
    }
    clearMobileVoiceError();

    if (hasSupport) {
      if (isListening) {
        stopListening();
      } else {
        startListening();
      }
      return;
    }

    if (isMobileRecording) {
      stopMobileRecording();
      return;
    }
    await startMobileRecording();
  }, [
    clearMobileVoiceError,
    hasSupport,
    isListening,
    isMobileRecording,
    onInterrupt,
    onVoiceInputStart,
    startListening,
    startMobileRecording,
    stopListening,
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
    <div className={`control-panel ${containerClass}`}>
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

      <div className="control-panel-actions mt-3 flex items-center justify-between text-xs text-slate-400">
        <span className="control-panel-count">{mode === 'chat' ? chatText.length : text.length} chars</span>
        <div className="control-panel-buttons flex items-center gap-2">
          {mode === 'chat' && canUseAnyVoice && (
            <button
              onClick={startListeningSafe}
              disabled={isDisabled || isMobileTranscribing}
              className={`rounded-lg px-3 py-2 text-xs font-bold ${
                isVoiceListening ? 'bg-rose-500 text-white' : 'bg-slate-800 text-slate-200'
              }`}
            >
              {isMobileTranscribing ? 'Transcribing…' : (isVoiceListening ? 'Listening…' : 'Voice Input')}
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
      {mode === 'chat' && !canUseAnyVoice && (
        <p className="mt-2 text-[11px] text-slate-500">
          Voice input is unavailable in this browser.
        </p>
      )}
      {mobileVoiceError && (
        <p className="mt-2 text-[11px] text-rose-400">
          {mobileVoiceError}
        </p>
      )}
    </div>
  );
};

export default ControlPanel;
