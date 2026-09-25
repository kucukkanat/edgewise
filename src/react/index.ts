/**
 * React hooks: one per verb, plus model state and the microphone.
 * @module
 */
import { createContext, createElement, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { type ConfigureOptions, configure } from '../core/config.ts';
import type { EdgewiseError } from '../core/errors.ts';
import { preload } from '../core/lifecycle.ts';
import type { AudioSource, Input, Message } from '../core/parts.ts';
import { registry } from '../core/registry.ts';
import type { Run } from '../core/run.ts';
import { capabilities, isRunnable, selectVariant, unload } from '../core/runtime.ts';
import type { Capabilities, CommonOptions, Device, ModelRef, PartType } from '../core/types.ts';
import { type Mic, type MicOptions, mic as openMic } from '../inputs/mic.ts';
import { embed } from '../verbs/embed.ts';
import { type EvaluateResult, evaluate, type Question } from '../verbs/evaluate.ts';
import { type ForecastResult, forecast, type SeriesInput } from '../verbs/forecast.ts';
import { type GenerateOptions, type GenerateResult, generate } from '../verbs/generate.ts';
import { type PaintOptions, type PaintResult, paint } from '../verbs/paint.ts';
import { type SpeakOptions, speak } from '../verbs/speak.ts';

const Ctx = createContext<{ ready: boolean }>({ ready: true });

/** Apply Edgewise settings for the tree below. */
export function EdgewiseProvider(props: ConfigureOptions & { children?: ReactNode }): ReactNode {
  const { children, ...cfg } = props;
  const key = JSON.stringify(cfg);
  useMemo(() => configure(cfg), [key]);
  return createElement(Ctx.Provider, { value: { ready: true } }, children);
}

/** Device capabilities, or null while they load. */
export function useCapabilities(): Capabilities | null {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  useEffect(() => {
    let live = true;
    capabilities().then((c) => live && setCaps(c));
    return () => {
      live = false;
    };
  }, []);
  return caps;
}

export type ModelStatus = 'idle' | 'downloading' | 'compiling' | 'ready' | 'error' | 'unsupported';

export interface ModelState {
  status: ModelStatus;
  /** 0 to 1 while downloading. */
  progress: number;
  /** Download size of the variant this device would use. */
  sizeBytes: number | null;
  accepts: PartType[];
  device: Device | null;
  error: EdgewiseError | null;
  load(): Promise<void>;
  unload(): Promise<void>;
}

/** Download state for a model, and buttons' worth of load/unload. */
export function useModel(id: string, o: { autoLoad?: boolean; allowPreview?: boolean } = {}): ModelState {
  const m = useMemo(() => registry.get(id), [id]);
  const [status, setStatus] = useState<ModelStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [sizeBytes, setSize] = useState<number | null>(null);
  const [device, setDevice] = useState<Device | null>(null);
  const [error, setError] = useState<EdgewiseError | null>(null);
  useEffect(() => {
    let live = true;
    (async () => {
      const caps = await capabilities();
      if (!isRunnable(m, caps)) {
        if (live) setStatus('unsupported');
        return;
      }
      const sel = await selectVariant(m).catch(() => null);
      if (!live || !sel) return;
      setSize(sel.variant.bytes ?? null);
      setDevice(sel.device);
    })();
    return () => {
      live = false;
    };
  }, [m]);
  const load = useCallback(async () => {
    setStatus('downloading');
    setError(null);
    const files = new Map<string, { loaded: number; total: number }>();
    try {
      await preload([m], {
        allowPreview: o.allowPreview,
        onProgress: (e) => {
          if (e.type === 'download') {
            files.set(e.file, { loaded: e.loaded, total: e.total });
            let l = 0;
            let t = 0;
            for (const f of files.values()) {
              l += f.loaded;
              t += f.total;
            }
            setProgress(t ? l / t : 0);
          } else if (e.type === 'compile') setStatus('compiling');
          else if (e.type === 'ready') setDevice(e.device);
        },
      });
      setProgress(1);
      setStatus('ready');
    } catch (err) {
      setError(err as EdgewiseError);
      setStatus('error');
    }
  }, [m, o.allowPreview]);
  useEffect(() => {
    if (o.autoLoad) void load();
  }, [o.autoLoad, load]);
  return {
    status,
    progress,
    sizeBytes,
    accepts: m.accepts,
    device,
    error,
    load,
    unload: async () => {
      await unload(m.id);
      setStatus('idle');
      setProgress(0);
    },
  };
}

type GenOpts = Omit<GenerateOptions, 'input' | 'messages' | 'signal'>;

/** Stream text from any generate model. `run(input)` starts a new generation. */
export function useGenerate(o: GenOpts & { input?: Input }) {
  const [text, setText] = useState('');
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [isRunning, setRunning] = useState(false);
  const [error, setError] = useState<EdgewiseError | null>(null);
  const ac = useRef<AbortController | null>(null);
  const optsRef = useRef(o);
  optsRef.current = o;
  const run = useCallback(async (input?: Input) => {
    ac.current?.abort();
    const ctrl = new AbortController();
    ac.current = ctrl;
    setText('');
    setError(null);
    setRunning(true);
    try {
      const { input: _i, ...rest } = optsRef.current;
      const r = generate({ ...rest, input: input ?? optsRef.current.input, signal: ctrl.signal });
      let acc = '';
      for await (const d of r) {
        if (typeof d === 'string') {
          acc += d;
          setText(acc);
        }
      }
      const res = await r;
      setResult(res);
      setText(res.text);
      return res;
    } catch (err) {
      if (!ctrl.signal.aborted) setError(err as EdgewiseError);
      return null;
    } finally {
      setRunning(false);
    }
  }, []);
  // A live audio source (such as useMic().source) starts transcription automatically.
  const src = o.input && typeof o.input === 'object' && (o.input as AudioSource).kind === 'audio-source' ? (o.input as AudioSource) : null;
  useEffect(() => {
    if (src) void run(src);
    return () => ac.current?.abort();
  }, [src, run]);
  return { text, result, isRunning, error, run, stop: () => ac.current?.abort() };
}

export interface ChatMessage extends Message {
  id: string;
}

/** A chat with any generate model. Attachments work when the model accepts them. */
export function useChat(o: GenOpts) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Blob[]>([]);
  const [isStreaming, setStreaming] = useState(false);
  const [error, setError] = useState<EdgewiseError | null>(null);
  const ac = useRef<AbortController | null>(null);
  const seq = useRef(0);
  const optsRef = useRef(o);
  optsRef.current = o;
  const submit = useCallback(
    async (text?: string) => {
      const content = text ?? input;
      if (!content.trim() && !attachments.length) return;
      const user: ChatMessage = {
        id: `m${++seq.current}`,
        role: 'user',
        content: attachments.length ? [...attachments.map((b) => ({ type: 'image' as const, image: b })), { type: 'text' as const, text: content }] : content,
      };
      const history = [...messages, user];
      setMessages(history);
      setInput('');
      setAttachments([]);
      setStreaming(true);
      setError(null);
      const ctrl = new AbortController();
      ac.current = ctrl;
      const reply: ChatMessage = { id: `m${++seq.current}`, role: 'assistant', content: '' };
      try {
        const r = generate({ ...optsRef.current, messages: history.map(({ id: _id, ...m }) => m), signal: ctrl.signal });
        let acc = '';
        for await (const d of r) {
          if (typeof d !== 'string') continue;
          acc += d;
          setMessages([...history, { ...reply, content: acc }]);
        }
        const res = await r;
        setMessages([...history, { ...reply, content: res.text, toolCalls: res.toolCalls.length ? res.toolCalls : undefined }]);
      } catch (err) {
        if (!ctrl.signal.aborted) setError(err as EdgewiseError);
      } finally {
        setStreaming(false);
      }
    },
    [input, attachments, messages],
  );
  return {
    messages,
    setMessages,
    input,
    setInput,
    attach: (b: Blob) => setAttachments((a) => [...a, b]),
    attachments,
    submit,
    stop: () => ac.current?.abort(),
    isStreaming,
    error,
  };
}

/** Evaluate questions whenever the state changes (debounced). */
export function useEvaluate<Q extends Record<string, Question>>(o: CommonOptions & { model?: ModelRef; state: unknown; questions: Q; debounceMs?: number }) {
  const [res, setRes] = useState<EvaluateResult<Q> | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<EdgewiseError | null>(null);
  const key = JSON.stringify(o.state ?? null);
  const optsRef = useRef(o);
  optsRef.current = o;
  useEffect(() => {
    if (o.state === undefined || o.state === '') return;
    let live = true;
    setPending(true);
    const t = setTimeout(async () => {
      try {
        const { debounceMs: _d, ...rest } = optsRef.current;
        const r = await evaluate(rest);
        if (live) {
          setRes(r);
          setError(null);
        }
      } catch (err) {
        if (live) setError(err as EdgewiseError);
      } finally {
        if (live) setPending(false);
      }
    }, o.debounceMs ?? 300);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [key, o.debounceMs]);
  return { answers: res?.answers ?? null, confidence: res?.confidence ?? null, pending, error };
}

/** Embed text on demand. */
export function useEmbed(o: { model: ModelRef; purpose?: 'query' | 'document'; dimensions?: number }) {
  const [pending, setPending] = useState(false);
  const optsRef = useRef(o);
  optsRef.current = o;
  const run = useCallback(async (text: string) => {
    setPending(true);
    try {
      return (await embed({ ...optsRef.current, input: text })).embedding;
    } finally {
      setPending(false);
    }
  }, []);
  return { embed: run, pending };
}

/** Speak text. `speak()` streams to the speakers; `stop()` cuts it off. */
export function useSpeak(o: Omit<SpeakOptions, 'input' | 'signal'>) {
  const [speaking, setSpeaking] = useState(false);
  const ac = useRef<AbortController | null>(null);
  const optsRef = useRef(o);
  optsRef.current = o;
  const say = useCallback(async (input: string | AsyncIterable<string>) => {
    ac.current?.abort();
    const ctrl = new AbortController();
    ac.current = ctrl;
    setSpeaking(true);
    try {
      return await speak({ ...optsRef.current, input, signal: ctrl.signal }).play();
    } finally {
      setSpeaking(false);
    }
  }, []);
  return { speak: say, speaking, stop: () => ac.current?.abort() };
}

/** Paint images with step previews. */
export function usePaint(o: Omit<PaintOptions, 'prompt' | 'signal'>) {
  const [result, setResult] = useState<PaintResult | null>(null);
  const [preview, setPreview] = useState<ImageData | null>(null);
  const [isRunning, setRunning] = useState(false);
  const ac = useRef<AbortController | null>(null);
  const optsRef = useRef(o);
  optsRef.current = o;
  const run = useCallback(async (prompt: string) => {
    ac.current?.abort();
    const ctrl = new AbortController();
    ac.current = ctrl;
    setRunning(true);
    try {
      const r = paint({ ...optsRef.current, prompt, signal: ctrl.signal });
      for await (const s of r)
        if (typeof ImageData !== 'undefined') setPreview(new ImageData(s.preview.data as Uint8ClampedArray<ArrayBuffer>, s.preview.width, s.preview.height));
      const res = await r;
      setResult(res);
      return res;
    } finally {
      setRunning(false);
    }
  }, []);
  return { run, result, preview, isRunning, stop: () => ac.current?.abort() };
}

/** Forecast a series whenever it changes. */
export function useForecast(o: CommonOptions & { model: ModelRef; series: SeriesInput | null; horizon: number; quantiles?: number[] }) {
  const [res, setRes] = useState<ForecastResult | null>(null);
  const [error, setError] = useState<EdgewiseError | null>(null);
  const key = o.series ? `${o.series.length}:${JSON.stringify(Array.from(o.series as ArrayLike<unknown>).slice(-8))}:${o.horizon}` : '';
  const optsRef = useRef(o);
  optsRef.current = o;
  useEffect(() => {
    if (!o.series) return;
    let live = true;
    forecast({ ...optsRef.current, series: optsRef.current.series as SeriesInput })
      .then((r) => live && setRes(r))
      .catch((e) => live && setError(e));
    return () => {
      live = false;
    };
  }, [key]);
  return { forecast: res, error };
}

/** The microphone as React state. Pass `source` as the input of a speech model. */
export function useMic(o: MicOptions = {}) {
  const [m, setMic] = useState<Mic | null>(null);
  const [listening, setListening] = useState(false);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<EdgewiseError | null>(null);
  useEffect(() => {
    if (!listening || !m) return;
    let raf = 0;
    const tick = () => {
      setLevel(m.level);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [listening, m]);
  useEffect(() => () => m?.dispose(), [m]);
  const key = JSON.stringify(o);
  const start = useCallback(async () => {
    try {
      const x = m ?? (await openMic(JSON.parse(key)));
      setMic(x);
      await x.start();
      setListening(true);
    } catch (err) {
      setError(err as EdgewiseError);
    }
  }, [m, key]);
  const stop = useCallback(async () => {
    setListening(false);
    return (await m?.stop()) ?? new Float32Array(0);
  }, [m]);
  return { start, stop, listening, level, error, source: listening ? m : null };
}

export const _ctx = Ctx;
export function useEdgewise() {
  return useContext(Ctx);
}
export type { Run };
