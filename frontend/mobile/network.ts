const DEFAULT_BACKEND_PORT = 8000;

export const normalizeApiBase = (value: string) => value.trim().replace(/\/+$/, '');

const inferDefaultApiBase = () => {
  if (typeof window === 'undefined') {
    return `http://127.0.0.1:${DEFAULT_BACKEND_PORT}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  const host = window.location.hostname || '127.0.0.1';
  return `${protocol}//${host}:${DEFAULT_BACKEND_PORT}`;
};

export const getDefaultApiBase = () => {
  const envApiBase = (import.meta.env.VITE_API_BASE || '').trim();
  return normalizeApiBase(envApiBase || inferDefaultApiBase());
};
