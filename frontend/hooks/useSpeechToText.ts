import { useEffect, useRef, useState } from 'react';

type SpeechResultHandler = (text: string) => void;

export const useSpeechToText = (onFinalText: SpeechResultHandler) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [hasSupport, setHasSupport] = useState(false);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setHasSupport(false);
      recognitionRef.current = null;
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = true;

    recognition.onresult = (event: any) => {
      const current = event.resultIndex;
      const text = event.results[current][0].transcript;
      setTranscript(text);
      if (event.results[current].isFinal) {
        setIsListening(false);
        onFinalText(text);
      }
    };

    recognition.onerror = () => {
      setIsListening(false);
    };
    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;
    setHasSupport(true);

    return () => {
      try {
        recognition.stop();
      } catch {
        // ignore cleanup errors
      }
      recognitionRef.current = null;
      setHasSupport(false);
    };
  }, [onFinalText]);

  const startListening = () => {
    if (!recognitionRef.current || isListening) return;
    try {
      setTranscript('');
      recognitionRef.current.start();
      setIsListening(true);
    } catch {
      // ignore duplicate starts
    }
  };

  const stopListening = () => {
    if (!recognitionRef.current) return;
    try {
      recognitionRef.current.stop();
    } catch {
      // ignore duplicate stops
    } finally {
      setIsListening(false);
    }
  };

  return {
    isListening,
    transcript,
    startListening,
    stopListening,
    hasSupport,
  };
};
