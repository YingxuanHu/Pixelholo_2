import React, { useCallback, useRef, useState } from 'react';
import { useSpeechToText } from '../hooks/useSpeechToText';

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
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcribeError, setTranscribeError] = useState<string | null>(null);
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

  const { isListening, startListening, hasSupport, transcript } = useSpeechToText(onSpeechResult);
  const canUseNativeSpeech = hasSupport && isSecureLikeContext;

  const startListeningSafe = useCallback(async () => {
    if (onInterrupt) {
      await onInterrupt();
    }
    startListening();
  }, [onInterrupt, startListening]);

  const openRecorder = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleRecordedAudio = useCallback(
    async (file: File | null) => {
      if (!file || !apiBase) return;
      setTranscribeError(null);
      setIsTranscribing(true);
      try {
        const form = new FormData();
        form.append('file', file);
        form.append('language', 'en');

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
        setChatText(spoken);
        await handleSend('chat', spoken);
      } catch (err) {
        setTranscribeError(err instanceof Error ? err.message : 'Transcription failed');
      } finally {
        setIsTranscribing(false);
      }
    },
    [apiBase, handleSend],
  );

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
          {mode === 'chat' && canUseNativeSpeech && (
            <button
              onClick={startListeningSafe}
              disabled={isDisabled}
              className={`rounded-lg px-3 py-2 text-xs font-bold ${
                isListening ? 'bg-rose-500 text-white' : 'bg-slate-800 text-slate-200'
              }`}
            >
              {isListening ? 'Listening…' : 'Voice Input'}
            </button>
          )}
          {mode === 'chat' && !canUseNativeSpeech && apiBase && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                capture="user"
                className="hidden"
                onChange={async (e) => {
                  const file = e.target.files?.[0] || null;
                  await handleRecordedAudio(file);
                  e.currentTarget.value = '';
                }}
              />
              <button
                onClick={openRecorder}
                disabled={isDisabled || isTranscribing}
                className="rounded-lg px-3 py-2 text-xs font-bold bg-slate-800 text-slate-200"
              >
                {isTranscribing ? 'Transcribing…' : 'Record Clip'}
              </button>
            </>
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

      {mode === 'chat' && !canUseNativeSpeech && (
        <p className="mt-2 text-[11px] text-slate-500">
          Native browser speech recognition is unavailable on this device/context. Use <span className="font-semibold text-slate-300">Record Clip</span>.
        </p>
      )}
      {transcribeError && (
        <p className="mt-2 text-[11px] text-rose-400">{transcribeError}</p>
      )}
    </div>
  );
};

export default ControlPanel;
