/** The six public verbs, plus `vad`, which powers `mic()` and is not a verb. */
export type Verb = 'generate' | 'evaluate' | 'embed' | 'speak' | 'paint' | 'forecast' | 'vad';

export type PartType = 'text' | 'image' | 'audio' | 'video' | 'series';

export type Status = 'stable' | 'preview' | 'experimental';

/** Where inference runs. `wasm` is the CPU path in browsers, `cpu` the native CPU path in Bun and Node. */
export type Device = 'webgpu' | 'wasm' | 'cpu';
export type DeviceOption = 'auto' | Device;

export type Dtype = 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'q4f16' | 'bnb4';

/** Internal pipeline kind. Decides which backend code runs a model. */
export type Task =
  | 'causal-lm'
  | 'image-text-to-text'
  | 'florence2'
  | 'speech-to-text'
  | 'feature-extraction'
  | 'zero-shot-nli'
  | 'lfm-router'
  | 'sequence-classification'
  | 'token-classification'
  | 'lfm-token-classification'
  | 'kokoro'
  | 'sd-turbo'
  | 'chronos-bolt'
  | 'silero-vad'
  | 'chrome-prompt'
  | 'mock';

export interface Variant {
  /** One dtype for the whole model, or one per ONNX component. */
  dtype: Dtype | Record<string, Dtype>;
  /** Devices this variant is known to run correctly on. */
  devices: Device[];
  /** Approximate download size in bytes. */
  bytes?: number;
  /** Requires the WebGPU `shader-f16` feature. */
  shaderF16?: boolean;
}

export type ModelSource =
  /** A Hugging Face repo, pinned to a commit. */
  | { repo: string; revision: string; subfolder?: string }
  /** Files in a GitHub repo, pinned to a commit, served from raw.githubusercontent.com. */
  | { github: string; revision: string; path: string }
  /** A Hugging Face storage bucket. Buckets are not versioned, so pin files with `config.sha256`. */
  | { bucket: string; path: string }
  /** Any base URL. Files are fetched relative to it. */
  | { url: string }
  | { builtin: 'chrome' };

export interface Manifest {
  id: string;
  /** Manifest version. Bumped when the pinned revision or variants change. */
  version: string;
  verb: Verb;
  accepts: PartType[];
  task: Task;
  source: ModelSource;
  variants: Variant[];
  requires?: { webgpu?: boolean; browser?: boolean; crossOriginIsolated?: boolean };
  /** Human-readable parameter count, such as "350M". */
  params?: string;
  license: string;
  status: Status;
  description?: string;
  features?: string[];
  /** Task-specific settings, such as tool-call format, pooling or labels. */
  config?: Record<string, unknown>;
}

/** What `defineModel` accepts. Missing fields get sensible defaults. */
export type ManifestInput = Omit<Manifest, 'version' | 'status' | 'license' | 'variants'> &
  Partial<Pick<Manifest, 'version' | 'status' | 'license' | 'variants'>>;

export type LoadEvent =
  | { type: 'download'; model: string; file: string; loaded: number; total: number }
  | { type: 'compile'; model: string }
  | { type: 'ready'; model: string; device: Device; dtype: string };

/** Options shared by every verb. */
export interface CommonOptions {
  /** Abort download or inference. */
  signal?: AbortSignal;
  /** Receives load events while the model downloads and compiles. */
  onProgress?: (event: LoadEvent) => void;
  /** Force a device. Default `'auto'`. */
  device?: DeviceOption;
  /** Force a dtype, such as `'q4'`. */
  dtype?: Dtype | Record<string, Dtype>;
  /** Allow preview and experimental models. */
  allowPreview?: boolean;
}

export type ModelRef = string | Manifest;
export type ModelSpec = ModelRef | ModelRef[];

export interface Capabilities {
  runtime: 'browser' | 'bun' | 'node' | 'deno' | 'worker';
  webgpu: boolean;
  shaderF16: boolean;
  /** Name of the GPU adapter, when one was found. */
  adapter: string | null;
  /** `false` when the only adapter is a CPU software renderer. */
  hardwareGpu: boolean;
  /** Where the GPU device comes from. */
  gpuProvider: 'navigator' | 'vgpu' | null;
  crossOriginIsolated: boolean;
  threads: boolean;
  builtinAI: 'available' | 'downloadable' | 'unavailable';
  cores: number | null;
  storage: { quota: number | null; usage: number | null };
  tier: 'gpu-high' | 'gpu' | 'mobile' | 'cpu';
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  tokensPerSecond: number;
}

/** Which backend actually ran a call. Returned on every result. */
export interface RunInfo {
  model: string;
  device: Device;
  dtype: string;
  backend: string;
}
