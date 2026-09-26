import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, rename, rm, stat, statfs, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { debug, getConfig } from '../core/config.ts';
import { AbortError, DownloadError, toEdgewiseError, UnsupportedDeviceError } from '../core/errors.ts';
import { decodeWav, isWav } from './audio.ts';
import type { DecodedAudio, FetchOptions, GpuInfo, OrtModule, Platform } from './types.ts';

const require = createRequire(import.meta.url);

const runtime: Platform['name'] =
  typeof (globalThis as { Bun?: unknown }).Bun !== 'undefined' ? 'bun' : typeof (globalThis as { Deno?: unknown }).Deno !== 'undefined' ? 'deno' : 'node';

export function cacheRoot(): string {
  return getConfig().cacheDir ?? process.env.EDGEWISE_CACHE ?? path.join(os.homedir(), '.cache', 'edgewise');
}

let gpuPromise: Promise<GpuInfo> | null = null;

/** Linux GPUs are reached through Vulkan. Without an ICD manifest there is no GPU to find. */
function hasVulkanDriver(): boolean {
  if (process.env.VK_ICD_FILENAMES || process.env.VK_DRIVER_FILES) return true;
  const { existsSync, readdirSync } = require('node:fs') as typeof import('node:fs');
  for (const d of ['/usr/share/vulkan/icd.d', '/etc/vulkan/icd.d', '/usr/local/share/vulkan/icd.d']) {
    try {
      if (existsSync(d) && readdirSync(d).some((f) => f.endsWith('.json'))) return true;
    } catch {
      // unreadable
    }
  }
  return false;
}

async function detect(): Promise<GpuInfo> {
  const none: GpuInfo = { webgpu: false, shaderF16: false, adapter: null, hardware: false, provider: null };
  const mode = process.env.EDGEWISE_GPU === 'force' ? 'force' : getConfig().serverGpu;
  if (mode === 'off' || process.env.EDGEWISE_GPU === 'off') return none;
  if (mode !== 'force' && process.platform === 'linux' && !hasVulkanDriver()) {
    debug('no Vulkan driver found; skipping GPU probe');
    return none;
  }
  try {
    const { createNodeAdapter } = (await import('vgpu/node')) as {
      createNodeAdapter: (o: { adapter: 'hardware' | 'software' | 'auto' }) => {
        requestDevice: (o?: Record<string, unknown>) => Promise<{
          gpu: GPUDevice;
          adapterInfo?: GPUAdapterInfo | null;
          features?: ReadonlySet<string>;
        }>;
      };
    };
    const want = mode === 'force' ? 'auto' : 'hardware';
    const device = await createNodeAdapter({ adapter: want }).requestDevice();
    const info = (device.adapterInfo ?? {}) as { description?: string; vendor?: string; device?: string; adapterType?: string };
    const name = info.description || info.device || info.vendor || 'unknown adapter';
    const software = info.adapterType === 'cpu' || /llvmpipe|lavapipe|swiftshader/i.test(name);
    const features = device.gpu?.features as ReadonlySet<string> | undefined;
    const f16 = !!features?.has('shader-f16');
    // Release the probe device right away: an open Dawn device keeps Node and Bun from exiting.
    try {
      (device as unknown as { destroy?: () => void }).destroy?.();
      device.gpu?.destroy?.();
    } catch {
      // already gone
    }
    const hardware = !software;
    debug('vgpu adapter:', name, hardware ? '(hardware)' : '(software)');
    return {
      webgpu: hardware || mode === 'force',
      shaderF16: f16,
      adapter: name,
      hardware,
      provider: 'vgpu',
    };
  } catch (err) {
    debug('no server GPU:', err instanceof Error ? err.message : err);
    return none;
  }
}

function keyFor(url: string): string {
  const h = createHash('sha256').update(url).digest('hex').slice(0, 16);
  let slug = 'file';
  try {
    // Keep the whole path readable (org, repo, revision, file) so cache.delete(repo) can match it.
    slug = new URL(url).pathname.split('/').filter(Boolean).join('_') || 'file';
  } catch {
    // not a URL
  }
  return `${h}-${slug.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 180)}`;
}

interface Inflight {
  promise: Promise<string>;
  controller: AbortController;
  waiters: number;
  progress: Set<(loaded: number, total: number) => void>;
  last?: [number, number];
}

const inflight = new Map<string, Inflight>();

/**
 * Download a file into the cache (streaming to disk) and return its path.
 * Concurrent callers share one download; each caller's signal only cancels its own wait,
 * and the download stops when every caller has given up.
 */
export function fetchToPath(url: string, opts: FetchOptions = {}): Promise<string> {
  let entry = inflight.get(url);
  if (!entry) {
    const e: Inflight = { controller: new AbortController(), waiters: 0, progress: new Set(), promise: Promise.resolve('') };
    e.promise = download(url, {
      sha256: opts.sha256,
      signal: e.controller.signal,
      onProgress: (l, t) => {
        e.last = [l, t];
        for (const fn of e.progress) fn(l, t);
      },
    }).finally(() => inflight.delete(url));
    e.promise.catch(() => {});
    inflight.set(url, e);
    entry = e;
  }
  const shared = entry;
  shared.waiters++;
  if (opts.onProgress) {
    shared.progress.add(opts.onProgress);
    if (shared.last) opts.onProgress(...shared.last);
  }
  return new Promise<string>((resolve, reject) => {
    let settled = false;
    const leave = () => {
      if (settled) return false;
      settled = true;
      shared.waiters--;
      if (opts.onProgress) shared.progress.delete(opts.onProgress);
      opts.signal?.removeEventListener('abort', onAbort);
      return true;
    };
    const onAbort = () => {
      if (!leave()) return;
      reject(new AbortError(undefined, { cause: opts.signal?.reason }));
      if (shared.waiters === 0) shared.controller.abort(opts.signal?.reason);
    };
    if (opts.signal?.aborted) {
      onAbort();
      return;
    }
    opts.signal?.addEventListener('abort', onAbort, { once: true });
    shared.promise.then(
      (v) => leave() && resolve(v),
      (err) => leave() && reject(err),
    );
  });
}

async function download(url: string, opts: FetchOptions): Promise<string> {
  const dir = path.join(cacheRoot(), 'files');
  const file = path.join(dir, keyFor(url));
  try {
    const s = await stat(file);
    opts.onProgress?.(s.size, s.size);
    return file;
  } catch {
    // not cached
  }
  await mkdir(dir, { recursive: true });
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
  const total = Number(res.headers.get('content-length')) || 0;
  const tmp = `${file}.${process.pid}.${Date.now()}.part`;
  const { createWriteStream } = await import('node:fs');
  const out = createWriteStream(tmp);
  // Surface open/write failures (disk full, permissions) as rejections instead of uncaught exceptions.
  const failed = new Promise<never>((_, reject) => out.once('error', reject));
  failed.catch(() => {});
  const hash = opts.sha256 ? createHash('sha256') : null;
  let loaded = 0;
  try {
    if (res.body) {
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.byteLength;
        hash?.update(value);
        if (!out.write(value)) await Promise.race([new Promise<void>((r) => out.once('drain', () => r())), failed]);
        opts.onProgress?.(loaded, total || loaded);
      }
    }
    await Promise.race([new Promise<void>((resolve, reject) => out.end((err?: Error | null) => (err ? reject(err) : resolve()))), failed]);
    if (total && loaded !== total) throw new DownloadError(`Download of ${url} ended early (${loaded} of ${total} bytes).`);
    if (hash && opts.sha256) {
      const got = hash.digest('hex');
      if (got !== opts.sha256.toLowerCase()) {
        throw new DownloadError(`Checksum mismatch for ${url}: expected ${opts.sha256}, got ${got}.`, {
          retryable: true,
          hint: 'The file was corrupted in transit, or the model file on the server changed. Update Edgewise, or report this if you are on the latest version.',
        });
      }
    }
    await rename(tmp, file);
  } catch (err) {
    out.destroy();
    await rm(tmp, { force: true });
    throw toEdgewiseError(err, `Download of ${url} failed`);
  }
  return file;
}

async function fetchCached(url: string, opts: FetchOptions = {}): Promise<Uint8Array> {
  const file = await fetchToPath(url, opts);
  const bytes = await readFile(file);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

async function walk(dir: string, out: { key: string; bytes: number }[], root: string): Promise<void> {
  let entries: import('node:fs').Dirent[];
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) await walk(p, out, root);
    else if (e.isFile()) out.push({ key: path.relative(root, p), bytes: (await stat(p)).size });
  }
}

let ortPromise: Promise<OrtModule> | null = null;

export const platform: Platform = {
  name: runtime,
  isBrowser: false,
  detectGpu() {
    gpuPromise ??= detect();
    return gpuPromise;
  },
  async storageEstimate() {
    try {
      const root = cacheRoot();
      await mkdir(root, { recursive: true });
      const s = await statfs(root);
      const list: { key: string; bytes: number }[] = [];
      await walk(root, list, root);
      return { quota: Number(s.bavail) * Number(s.bsize), usage: list.reduce((a, b) => a + b.bytes, 0) };
    } catch {
      return { quota: null, usage: null };
    }
  },
  async persist() {
    return true;
  },
  fetchCached,
  fetchModel: (url, opts) => fetchToPath(url, opts),
  async cacheList() {
    const root = cacheRoot();
    const out: { key: string; bytes: number }[] = [];
    await walk(root, out, root);
    return out;
  },
  async cacheDelete(prefix?: string) {
    const root = cacheRoot();
    const list: { key: string; bytes: number }[] = [];
    await walk(root, list, root);
    // Saved voices and vector indexes are user data, not model files: never delete them here.
    const victims = list.filter(
      (f) => !f.key.startsWith('indexes') && (!prefix || f.key.includes(prefix.replace(/[^A-Za-z0-9._-]/g, '_')) || f.key.includes(prefix)),
    );
    for (const f of victims) await rm(path.join(root, f.key), { force: true });
    return victims.length;
  },
  async decodeAudio(data: Uint8Array, mime?: string): Promise<DecodedAudio> {
    if (isWav(data) || mime === 'audio/wav' || mime === 'audio/x-wav') return decodeWav(data);
    throw new UnsupportedDeviceError(`Cannot decode ${mime || 'this audio format'} on ${runtime}.`, {
      hint: 'On Bun and Node, pass WAV audio or a Float32Array of 16 kHz samples. Decode MP3, Ogg or WebM first, for example with ffmpeg.',
    });
  },
  async store(file: string, data?: Uint8Array) {
    const dir = path.join(cacheRoot(), 'indexes');
    const f = path.join(dir, file);
    if (data) {
      await mkdir(dir, { recursive: true });
      await writeFile(f, data);
      return data;
    }
    try {
      return new Uint8Array(await readFile(f));
    } catch {
      return null;
    }
  },
  transformersCacheDir() {
    return path.join(cacheRoot(), 'hf');
  },
  loadOrt() {
    ortPromise ??= import('onnxruntime-node').then((m) => ((m as { default?: OrtModule }).default ?? m) as OrtModule);
    return ortPromise;
  },
  async playAudio() {
    throw new UnsupportedDeviceError(`Audio playback is not available on ${runtime}.`, {
      hint: 'Use audio.toWav() and write the bytes to a file, or pipe audio.samples to your own player.',
    });
  },
  cores() {
    return os.availableParallelism?.() ?? os.cpus().length;
  },
  crossOriginIsolated() {
    return true;
  },
};
