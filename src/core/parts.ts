import { ConfigError, UnsupportedInputError } from './errors.ts';
import type { PartType } from './types.ts';

/** Anything Edgewise can read as an image. */
export type ImageLike = ImageBitmap | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | OffscreenCanvas | Blob | URL | RawPixels;

/** Decoded RGB or RGBA pixels. Works in every runtime. */
export interface RawPixels {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
  channels: 1 | 3 | 4;
}

/** Anything Edgewise can read as audio. Float32Array is mono PCM at 16 kHz unless a sampleRate is given. */
export type AudioLike = Float32Array | AudioBuffer | Blob | URL | ArrayBuffer | Uint8Array;

export type Part =
  | { type: 'text'; text: string }
  | { type: 'image'; image: ImageLike }
  | { type: 'audio'; audio: AudioLike; sampleRate?: number }
  | { type: 'video'; video: HTMLVideoElement | Blob | ImageLike[]; frames?: number; fps?: number };

/** A live audio source, such as the one `mic()` returns. */
export interface AudioSource {
  readonly kind: 'audio-source';
  readonly sampleRate: number;
  /** Yields one Float32Array per utterance (with VAD) or per chunk (without). */
  utterances(): AsyncIterable<Float32Array>;
}

export type InputItem = string | Part | ImageLike | AudioLike;
export type Input = InputItem | InputItem[] | AudioSource;

export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | Part[];
  /** Tool calls made by the assistant in this message. */
  toolCalls?: { id: string; name: string; input: unknown }[];
  /** For role 'tool': which call this result answers. */
  toolCallId?: string;
  name?: string;
}

const hasGlobal = (name: string): boolean => typeof (globalThis as Record<string, unknown>)[name] === 'function';

function isInstance(value: unknown, ctor: string): boolean {
  const C = (globalThis as Record<string, unknown>)[ctor];
  return typeof C === 'function' && value instanceof (C as new (...a: never[]) => unknown);
}

export function isAudioSource(v: unknown): v is AudioSource {
  return typeof v === 'object' && v !== null && (v as AudioSource).kind === 'audio-source';
}

export function isRawPixels(v: unknown): v is RawPixels {
  if (typeof v !== 'object' || v === null) return false;
  const r = v as RawPixels;
  return (
    (r.data instanceof Uint8ClampedArray || r.data instanceof Uint8Array) &&
    typeof r.width === 'number' &&
    typeof r.height === 'number' &&
    (r.channels === 1 || r.channels === 3 || r.channels === 4)
  );
}

function isPart(v: unknown): v is Part {
  if (typeof v !== 'object' || v === null) return false;
  const t = (v as { type?: unknown }).type;
  return (
    (t === 'text' && typeof (v as { text?: unknown }).text === 'string') ||
    (t === 'image' && 'image' in v) ||
    (t === 'audio' && 'audio' in v) ||
    (t === 'video' && 'video' in v)
  );
}

function blobKind(b: Blob): PartType | null {
  const t = b.type || '';
  if (t.startsWith('image/')) return 'image';
  if (t.startsWith('audio/')) return 'audio';
  if (t.startsWith('video/')) return 'video';
  return null;
}

function urlKind(u: URL): PartType | null {
  const p = u.pathname.toLowerCase();
  if (/\.(png|jpe?g|gif|webp|bmp|avif)$/.test(p)) return 'image';
  if (/\.(wav|mp3|ogg|oga|flac|m4a|aac|opus|webm)$/.test(p)) return 'audio';
  if (/\.(mp4|mov|mkv)$/.test(p)) return 'video';
  return null;
}

/** Turn one input item into a Part. */
export function toPart(item: InputItem): Part {
  if (typeof item === 'string') return { type: 'text', text: item };
  if (isPart(item)) return item;
  if (item instanceof Float32Array) return { type: 'audio', audio: item };
  if (isInstance(item, 'AudioBuffer')) return { type: 'audio', audio: item as AudioBuffer };
  if (isRawPixels(item)) return { type: 'image', image: item };
  if (
    isInstance(item, 'HTMLImageElement') ||
    isInstance(item, 'HTMLCanvasElement') ||
    isInstance(item, 'ImageBitmap') ||
    isInstance(item, 'ImageData') ||
    isInstance(item, 'OffscreenCanvas') ||
    isInstance(item, 'HTMLVideoElement')
  ) {
    return { type: 'image', image: item as ImageLike };
  }
  if (isInstance(item, 'Blob')) {
    const kind = blobKind(item as Blob);
    if (kind === 'image') return { type: 'image', image: item as Blob };
    if (kind === 'audio') return { type: 'audio', audio: item as Blob };
    if (kind === 'video') return { type: 'video', video: item as Blob };
    throw new UnsupportedInputError(`Cannot tell what kind of file this is (type "${(item as Blob).type}").`, {
      hint: "Wrap it in a part, such as { type: 'image', image: blob }, or give the Blob a MIME type.",
    });
  }
  if (item instanceof URL) {
    const kind = urlKind(item);
    if (kind === 'image') return { type: 'image', image: item };
    if (kind === 'audio') return { type: 'audio', audio: item };
    throw new UnsupportedInputError(`Cannot tell what kind of file ${item.href} is from its extension.`, {
      hint: "Wrap the URL in a part, such as { type: 'image', image: new URL(...) }.",
    });
  }
  if (item instanceof ArrayBuffer || item instanceof Uint8Array) {
    return { type: 'audio', audio: item };
  }
  throw new UnsupportedInputError(`Unsupported input: ${Object.prototype.toString.call(item)}.`, {
    hint: 'Pass a string, an image, audio as a Float32Array or Blob, or a part object.',
  });
}

export interface NormalizedInput {
  messages: Message[];
  types: PartType[];
  source?: AudioSource;
}

/** Normalize `input` or `messages` into messages plus the set of part types used. */
export function normalizeInput(opts: { input?: Input; messages?: Message[]; system?: string }): NormalizedInput {
  if (opts.input !== undefined && opts.messages !== undefined) {
    throw new ConfigError('Pass either input or messages, not both.');
  }
  if (opts.input === undefined && opts.messages === undefined) {
    throw new ConfigError('Pass input or messages.');
  }
  const messages: Message[] = [];
  if (opts.system) messages.push({ role: 'system', content: opts.system });
  let source: AudioSource | undefined;
  if (opts.input !== undefined) {
    if (isAudioSource(opts.input)) {
      source = opts.input;
      messages.push({ role: 'user', content: [] });
    } else {
      const items = Array.isArray(opts.input) ? opts.input : [opts.input];
      if (items.length === 0) throw new ConfigError('input is an empty array.');
      const parts = items.map(toPart);
      messages.push({ role: 'user', content: parts.length === 1 && parts[0].type === 'text' ? parts[0].text : parts });
    }
  } else {
    for (const m of opts.messages as Message[]) {
      messages.push({ ...m, content: typeof m.content === 'string' ? m.content : m.content.map(toPart) });
    }
  }
  const types = new Set<PartType>();
  for (const m of messages) {
    if (typeof m.content === 'string') {
      if (m.content.length) types.add('text');
    } else for (const p of m.content) types.add(p.type);
  }
  if (source) types.add('audio');
  return { messages, types: [...types], source };
}

/** Plain text of a message. Non-text parts are dropped. */
export function messageText(m: Message): string {
  if (typeof m.content === 'string') return m.content;
  return m.content
    .filter((p): p is Extract<Part, { type: 'text' }> => p.type === 'text')
    .map((p) => p.text)
    .join('\n');
}

export const _internal = { hasGlobal };
