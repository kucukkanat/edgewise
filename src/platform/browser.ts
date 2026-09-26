import { DownloadError, toEdgewiseError, UnsupportedDeviceError } from '../core/errors.ts';
import { decodeWav, isWav } from './audio.ts';
import type { DecodedAudio, FetchOptions, GpuInfo, OrtModule, Platform } from './types.ts';

const CACHE = 'edgewise-files-v1';
const inWorker = typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== 'undefined';

let gpuPromise: Promise<GpuInfo> | null = null;

async function detect(): Promise<GpuInfo> {
  const none: GpuInfo = { webgpu: false, shaderF16: false, adapter: null, hardware: false, provider: null };
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return none;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return none;
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const name = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ') || 'WebGPU adapter';
    const software = (adapter as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter === true || /swiftshader|llvmpipe|basic render/i.test(name);
    return {
      webgpu: true,
      shaderF16: adapter.features.has('shader-f16'),
      adapter: name,
      hardware: !software,
      provider: 'navigator',
    };
  } catch {
    return none;
  }
}

async function openCache(): Promise<Cache | null> {
  try {
    return typeof caches !== 'undefined' ? await caches.open(CACHE) : null;
  } catch {
    return null;
  }
}

async function readBody(res: Response, opts: FetchOptions): Promise<Uint8Array> {
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    opts.onProgress?.(loaded, total || loaded);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.byteLength;
  }
  return out;
}

let audioCtx: AudioContext | null = null;
let ortPromise: Promise<OrtModule> | null = null;

export const platform: Platform = {
  name: inWorker ? 'worker' : 'browser',
  isBrowser: true,
  detectGpu() {
    gpuPromise ??= detect();
    return gpuPromise;
  },
  async storageEstimate() {
    try {
      const e = await navigator.storage?.estimate?.();
      return { quota: e?.quota ?? null, usage: e?.usage ?? null };
    } catch {
      return { quota: null, usage: null };
    }
  },
  async persist() {
    try {
      return (await navigator.storage?.persist?.()) ?? false;
    } catch {
      return false;
    }
  },
  async fetchCached(url: string, opts: FetchOptions = {}) {
    const cache = await openCache();
    const hit = await cache?.match(url);
    if (hit) {
      const b = new Uint8Array(await hit.arrayBuffer());
      opts.onProgress?.(b.byteLength, b.byteLength);
      return b;
    }
    let res: Response;
    try {
      res = await fetch(url, { signal: opts.signal });
    } catch (err) {
      throw toEdgewiseError(err, `Download of ${url} failed`);
    }
    if (!res.ok) {
      throw new DownloadError(`Download of ${url} failed with HTTP ${res.status}.`, {
        retryable: res.status >= 500 || res.status === 429,
      });
    }
    const bytes = await readBody(res, opts);
    if (opts.sha256) {
      const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>));
      const got = Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('');
      if (got !== opts.sha256.toLowerCase()) {
        throw new DownloadError(`Checksum mismatch for ${url}: expected ${opts.sha256}, got ${got}.`, {
          retryable: true,
          hint: 'The file was corrupted in transit, or the model file on the server changed. Update Edgewise, or report this if you are on the latest version.',
        });
      }
    }
    try {
      await cache?.put(
        url,
        new Response(bytes as Uint8Array<ArrayBuffer>, {
          headers: { 'content-type': res.headers.get('content-type') ?? 'application/octet-stream', 'content-length': String(bytes.byteLength) },
        }),
      );
    } catch {
      // Quota or private mode: keep working without the cache.
    }
    return bytes;
  },
  fetchModel(url: string, opts?: FetchOptions) {
    return platform.fetchCached(url, opts);
  },
  async cacheList() {
    const out: { key: string; bytes: number }[] = [];
    for (const name of ['edgewise-files-v1', 'transformers-cache']) {
      try {
        const c = await caches.open(name);
        for (const req of await c.keys()) {
          const r = await c.match(req);
          // Avoid reading multi-GB bodies just to size them; fall back to a Blob, which browsers back on disk.
          const len = Number(r?.headers.get('content-length')) || (r ? (await r.blob()).size : 0);
          out.push({ key: req.url, bytes: len });
        }
      } catch {
        // ignore
      }
    }
    return out;
  },
  async cacheDelete(prefix?: string) {
    let n = 0;
    for (const name of ['edgewise-files-v1', 'transformers-cache']) {
      try {
        const c = await caches.open(name);
        for (const req of await c.keys()) {
          if (!prefix || req.url.includes(prefix)) {
            await c.delete(req);
            n++;
          }
        }
      } catch {
        // ignore
      }
    }
    return n;
  },
  async decodeAudio(data: Uint8Array, mime?: string): Promise<DecodedAudio> {
    if (isWav(data)) return decodeWav(data);
    const Ctx = (globalThis as { OfflineAudioContext?: typeof OfflineAudioContext }).OfflineAudioContext;
    if (!Ctx) {
      throw new UnsupportedDeviceError(`Cannot decode ${mime || 'this audio'} here: no Web Audio API.`, {
        hint: 'Decode audio on the main thread, or pass WAV or a Float32Array.',
      });
    }
    const ctx = new Ctx(1, 1, 16000);
    const buf = await ctx.decodeAudioData(data.slice().buffer as ArrayBuffer);
    const ch = buf.numberOfChannels;
    const out = new Float32Array(buf.length);
    for (let c = 0; c < ch; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < d.length; i++) out[i] += d[i] / ch;
    }
    return { samples: out, sampleRate: buf.sampleRate };
  },
  async store(file: string, data?: Uint8Array) {
    const root = await navigator.storage?.getDirectory?.();
    if (!root) return null;
    if (data) {
      const h = await root.getFileHandle(file, { create: true });
      const w = await (h as FileSystemFileHandle & { createWritable: () => Promise<FileSystemWritableFileStream> }).createWritable();
      await w.write(data as Uint8Array<ArrayBuffer>);
      await w.close();
      return data;
    }
    try {
      const h = await root.getFileHandle(file);
      return new Uint8Array(await (await h.getFile()).arrayBuffer());
    } catch {
      return null;
    }
  },
  transformersCacheDir() {
    return undefined;
  },
  loadOrt() {
    // Share one ONNX Runtime with Transformers.js, which also decides where the .wasm files come from.
    ortPromise ??= (async () => {
      const { getTransformers } = await import('../backends/transformers.ts');
      await getTransformers();
      const m = await import('onnxruntime-web/webgpu');
      return ((m as { default?: OrtModule }).default ?? m) as OrtModule;
    })();
    return ortPromise;
  },
  async playAudio(samples: Float32Array, sampleRate: number, signal?: AbortSignal) {
    if (signal?.aborted) return;
    const Ctx = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext;
    if (!Ctx) throw new UnsupportedDeviceError('Audio playback needs the Web Audio API, which is not available here.');
    audioCtx ??= new Ctx();
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    const buf = audioCtx.createBuffer(1, samples.length, sampleRate);
    buf.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    await new Promise<void>((resolve) => {
      src.onended = () => resolve();
      signal?.addEventListener('abort', () => {
        try {
          src.stop();
        } catch {
          // already stopped
        }
        resolve();
      });
      src.start();
    });
  },
  cores() {
    return navigator.hardwareConcurrency ?? null;
  },
  crossOriginIsolated() {
    return !!(globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated;
  },
};
