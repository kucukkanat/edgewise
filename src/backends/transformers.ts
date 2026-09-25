import { debug, getConfig, onConfigChange } from '../core/config.ts';
import { ConfigError, toEdgewiseError } from '../core/errors.ts';
import { registry } from '../core/registry.ts';
import { dtypeLabel, getPlatform, type LoadContext, loadCached, selectVariant } from '../core/runtime.ts';
import type { CommonOptions, Device, LoadEvent, Manifest, RunInfo } from '../core/types.ts';

type TJS = typeof import('@huggingface/transformers');

let tjsPromise: Promise<TJS> | null = null;

/** The ONNX Runtime Web build Transformers.js uses, under a self-hosted prefix. */
export function ortWasmPaths(prefix: string): { mjs: string; wasm: string } {
  const base = prefix.endsWith('/') ? prefix : `${prefix}/`;
  return { mjs: `${base}ort-wasm-simd-threaded.asyncify.mjs`, wasm: `${base}ort-wasm-simd-threaded.asyncify.wasm` };
}

function applyEnv(t: TJS): void {
  const p = getPlatform();
  const cfg = getConfig();
  t.env.remoteHost = cfg.hub.endsWith('/') ? cfg.hub : `${cfg.hub}/`;
  t.env.allowLocalModels = false;
  t.env.allowRemoteModels = true;
  const dir = p.transformersCacheDir();
  if (dir) t.env.cacheDir = dir;
  if (p.isBrowser) t.env.useBrowserCache = true;
  const onnx = t.env.backends?.onnx as { wasm?: { wasmPaths?: unknown } } | undefined;
  if (p.isBrowser && cfg.wasmPaths && onnx?.wasm) onnx.wasm.wasmPaths = ortWasmPaths(cfg.wasmPaths);
}

/** Repo → pinned revision for every model in the registry, including tokenizer-only repos. */
function pinnedRevisions(): Map<string, string> {
  const out = new Map<string, string>();
  for (const m of registry.list()) {
    if ('repo' in m.source) out.set(m.source.repo, m.source.revision);
    const tok = m.config?.tokenizer as { repo?: string; revision?: string } | undefined;
    if (tok?.repo && tok.revision) out.set(tok.repo, tok.revision);
  }
  return out;
}

/**
 * Transformers.js requests a few metadata files at `main` even when a revision is given.
 * Rewrite those requests to the pinned commit so every byte comes from the pinned revision.
 */
function pinFetch(t: TJS): void {
  const env = t.env as unknown as { fetch: typeof fetch; __edgewisePinned?: boolean };
  if (env.__edgewisePinned) return;
  const original = env.fetch;
  env.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    const m = /^(https?:\/\/[^/]+\/)(.+?)\/resolve\/main\//.exec(url);
    if (m) {
      const rev = pinnedRevisions().get(m[2]);
      if (rev) return original(url.replace('/resolve/main/', `/resolve/${rev}/`), init);
    }
    return original(input as RequestInfo, init);
  }) as typeof fetch;
  env.__edgewisePinned = true;
}

/** Load Transformers.js once and point it at Edgewise's hub and cache. */
export function getTransformers(): Promise<TJS> {
  tjsPromise ??= import('@huggingface/transformers').then((t) => {
    applyEnv(t);
    pinFetch(t);
    onConfigChange(() => applyEnv(t));
    return t;
  });
  return tjsPromise;
}

export function repoOf(m: Manifest): { repo: string; revision: string } {
  if (!('repo' in m.source)) throw new ConfigError(`"${m.id}" is not a Hugging Face model.`);
  return { repo: m.source.repo, revision: m.source.revision };
}

/** Map a Transformers.js progress callback onto Edgewise load events. */
export function progressCallback(model: string, emit: (e: LoadEvent) => void) {
  const totals = new Map<string, { loaded: number; total: number }>();
  return (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (p.status === 'progress' && p.file) {
      totals.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
      emit({ type: 'download', model, file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 });
    } else if (p.status === 'done' && p.file) {
      const t = totals.get(p.file);
      if (t) emit({ type: 'download', model, file: p.file, loaded: t.total || t.loaded, total: t.total || t.loaded });
    }
  };
}

export function tjsDevice(d: Device): 'webgpu' | 'wasm' | 'cpu' {
  return d;
}

export interface Loaded<T> {
  value: T;
  info: RunInfo;
}

/**
 * Select a variant and load a Transformers.js model through the shared LRU cache.
 * On GPU failure it retries on the CPU path, as the fallback policy allows.
 */
export async function loadTjs<T>(
  m: Manifest,
  opts: CommonOptions,
  loader: (t: TJS, base: { device: Device; dtype: unknown; revision: string; progress_callback: unknown }, ctx: LoadContext) => Promise<T>,
  dispose?: (v: T) => Promise<void> | void,
): Promise<Loaded<T>> {
  const t = await getTransformers();
  const selection = await selectVariant(m, opts);
  const emit = (e: LoadEvent) => opts.onProgress?.(e);
  const attempt = async (sel: typeof selection): Promise<Loaded<T>> => {
    const key = `${m.id}|${sel.device}|${dtypeLabel(sel.dtype)}`;
    const ctx: LoadContext = { manifest: m, selection: sel, progress: emit, signal: opts.signal };
    const value = await loadCached(
      key,
      ctx,
      async () => {
        const { revision } = repoOf(m);
        debug(`loading ${m.id} on ${sel.device} (${dtypeLabel(sel.dtype)})`);
        try {
          return await loader(t, { device: sel.device, dtype: sel.dtype, revision, progress_callback: progressCallback(m.id, emit) }, ctx);
        } catch (err) {
          throw toEdgewiseError(err, `Loading ${m.id}`);
        }
      },
      dispose,
    );
    return { value, info: { model: m.id, device: sel.device, dtype: dtypeLabel(sel.dtype), backend: backendName(sel.device) } };
  };
  try {
    return await attempt(selection);
  } catch (err) {
    const cpu: Device = getPlatform().isBrowser ? 'wasm' : 'cpu';
    if (selection.device === 'webgpu' && getConfig().fallback.onBackendError !== 'throw' && opts.device !== 'webgpu') {
      debug(`${m.id} failed on webgpu, retrying on ${cpu}:`, err instanceof Error ? err.message : err);
      const sel = await selectVariant(m, { ...opts, device: cpu });
      return attempt(sel);
    }
    throw err;
  }
}

export function backendName(d: Device): string {
  const p = getPlatform();
  if (p.isBrowser) return d === 'webgpu' ? 'onnxruntime-web:webgpu' : 'onnxruntime-web:wasm';
  return d === 'webgpu' ? 'onnxruntime-node:webgpu' : 'onnxruntime-node:cpu';
}
