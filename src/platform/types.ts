export interface GpuInfo {
  /** A WebGPU adapter is usable for inference. */
  webgpu: boolean;
  shaderF16: boolean;
  adapter: string | null;
  /** False when the only adapter is a CPU software renderer. */
  hardware: boolean;
  provider: 'navigator' | 'vgpu' | null;
}

export interface FetchOptions {
  onProgress?: (loaded: number, total: number) => void;
  signal?: AbortSignal;
  /** Expected SHA-256 (hex). The download fails, and nothing is cached, if it does not match. */
  sha256?: string;
}

export interface DecodedAudio {
  samples: Float32Array;
  sampleRate: number;
}

export interface Platform {
  name: 'browser' | 'bun' | 'node' | 'deno' | 'worker';
  isBrowser: boolean;
  detectGpu(): Promise<GpuInfo>;
  storageEstimate(): Promise<{ quota: number | null; usage: number | null }>;
  persist(): Promise<boolean>;
  /** Download a file once and keep it in the Edgewise cache. */
  fetchCached(url: string, opts?: FetchOptions): Promise<Uint8Array>;
  /** Like fetchCached, but on servers returns a file path so large models are not held in memory twice. */
  fetchModel(url: string, opts?: FetchOptions): Promise<Uint8Array | string>;
  cacheList(): Promise<{ key: string; bytes: number }[]>;
  cacheDelete(prefix?: string): Promise<number>;
  decodeAudio(data: Uint8Array, mime?: string): Promise<DecodedAudio>;
  /** Directory Transformers.js caches into (server), or undefined (browser uses the Cache API). */
  transformersCacheDir(): string | undefined;
  /** Small persistent files (vector indexes): OPFS in browsers, `<cache>/indexes` on servers. Reads when `data` is omitted. */
  store(file: string, data?: Uint8Array): Promise<Uint8Array | null>;
  /** Load ONNX Runtime for this platform. */
  loadOrt(): Promise<OrtModule>;
  /** Play mono PCM. Browser only; resolves when playback ends. */
  playAudio(samples: Float32Array, sampleRate: number, signal?: AbortSignal): Promise<void>;
  cores(): number | null;
  crossOriginIsolated(): boolean;
}

// Minimal structural type for the parts of ONNX Runtime we use.
export interface OrtTensor {
  readonly type: string;
  readonly data: ArrayLike<number | bigint> & { length: number };
  readonly dims: readonly number[];
}
export interface OrtSession {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}
export interface OrtModule {
  Tensor: new (type: string, data: unknown, dims: readonly number[]) => OrtTensor;
  InferenceSession: {
    create(model: Uint8Array | string, options?: Record<string, unknown>): Promise<OrtSession>;
  };
  env: Record<string, unknown>;
}
