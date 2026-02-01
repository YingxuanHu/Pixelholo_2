import React, { useCallback, useState } from 'react';
import { useSpeechToText } from '../hooks/useSpeechToText';

type Mode = 'chat' | 'tts';

type ControlPanelProps = {
  onSendChat: (text: string) => Promise<void> | void;
  onSendDirect: (text: string) => Promise<void> | void;
  disabled?: boolean;
};

const ControlPanel: React.FC<ControlPanelProps> = ({ onSendChat, onSendDirect, disabled }) => {
  const [mode, setMode] = useState<Mode>('tts');
  const [text, setText] = useState('');
  const [isSending, setIsSending] = useState(false);

  const handleSend = useCallback(
    async (selectedMode: Mode, inputText: string) => {
      if (!inputText.trim() || isSending || disabled) return;
      setIsSending(true);
      try {
        if (selectedMode === 'chat') {
          await onSendChat(inputText);
        } else {
          await onSendDirect(inputText);
        }
        setText('');
      } finally {
        setIsSending(false);
      }
    },
    [disabled, isSending, onSendChat, onSendDirect],
  );

  const onSpeechResult = useCallback(
    async (spoken: string) => {
      setText(spoken);
      await handleSend('chat', spoken);
    },
    [handleSend],
  );

  const { isListening, startListening, hasSupport, transcript } = useSpeechToText(onSpeechResult);

  return (
    <div className="w-full rounded-2xl border border-slate-800 bg-slate-950/90 p-4 backdrop-blur">
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
        value={mode === 'chat' && transcript ? transcript : text}
        onChange={(e) => setText(e.target.value)}
        placeholder={mode === 'chat' ? 'Ask the assistant…' : 'Type what to say…'}
        className="h-24 w-full rounded-xl border border-slate-800 bg-slate-900/70 p-3 text-sm text-white outline-none"
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend(mode, text);
          }
        }}
      />

      <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
        <span>{text.length} chars</span>
        <div className="flex items-center gap-2">
          {mode === 'chat' && hasSupport && (
            <button
              onClick={startListening}
              disabled={disabled || isSending}
              className={`rounded-lg px-3 py-2 text-xs font-bold ${
                isListening ? 'bg-rose-500 text-white' : 'bg-slate-800 text-slate-200'
              }`}
            >
              {isListening ? 'Listening…' : 'Voice Input'}
            </button>
          )}
          <button
            onClick={() => handleSend(mode, text)}
            disabled={disabled || isSending}
            className="rounded-lg bg-teal-600 px-4 py-2 text-xs font-bold text-white"
          >
            {isSending ? 'Processing…' : mode === 'chat' ? 'Send to LLM' : 'Generate'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
