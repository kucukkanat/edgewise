import { loadChatterbox } from '../backends/chatterbox.ts';
import { loadChronos } from '../backends/chronos.ts';
import { loadEmbedder } from '../backends/embedding.ts';
import { fetchModelFile } from '../backends/files.ts';
import { loadFlorence } from '../backends/florence.ts';
import { loadPipe } from '../backends/judge.ts';
import { loadKokoro } from '../backends/kokoro.ts';
import { loadLfm } from '../backends/lfm.ts';
import { loadLm, loadVlm } from '../backends/lm.ts';
import { loadAsr } from '../backends/stt.ts';
import { loadSilero } from '../backends/vad.ts';
import { ConfigError } from './errors.ts';
import { registry } from './registry.ts';
import { getPlatform } from './runtime.ts';
import type { CommonOptions, LoadEvent, Manifest, ModelRef } from './types.ts';

async function warm(m: Manifest, o: CommonOptions): Promise<void> {
  switch (m.task) {
    case 'causal-lm':
      await loadLm(m, o);
      return;
    case 'image-text-to-text':
      await loadVlm(m, o);
      return;
    case 'florence2':
      await loadFlorence(m, o);
      return;
    case 'speech-to-text':
      await loadAsr(m, o);
      return;
    case 'zero-shot-nli':
    case 'sequence-classification':
    case 'token-classification':
      await loadPipe(m, o);
      return;
    case 'lfm-router':
    case 'lfm-token-classification':
      await loadLfm(m, o);
      return;
    case 'feature-extraction':
      await loadEmbedder(m, o);
      return;
    case 'kokoro':
      await loadKokoro(m, o);
      return;
    case 'chatterbox':
      await loadChatterbox(m, o);
      return;
    case 'sd-turbo':
      // Download only: the three graphs are compiled one at a time when painting, to save memory.
      for (const part of ['text_encoder', 'unet', 'vae_decoder']) {
        await fetchModelFile(m, `${part}/model.onnx`, { signal: o.signal, onProgress: o.onProgress });
      }
      return;
    case 'chronos-bolt':
      await loadChronos(m, o);
      return;
    case 'silero-vad':
      await loadSilero(m, o.signal);
      return;
    case 'chrome-prompt': {
      const LM = (globalThis as { LanguageModel?: { create: (o?: unknown) => Promise<{ destroy?: () => void }> } }).LanguageModel;
      if (LM) (await LM.create())?.destroy?.();
      return;
    }
  }
}

export interface PreloadOptions {
  onProgress?: (e: LoadEvent) => void;
  signal?: AbortSignal;
  allowPreview?: boolean;
  device?: CommonOptions['device'];
}

/**
 * Download and load models ahead of time, so the first call is fast.
 * Models load one after another to keep memory use predictable.
 */
export async function preload(models: ModelRef[], o: PreloadOptions = {}): Promise<void> {
  if (!Array.isArray(models)) throw new ConfigError('preload() takes an array of model IDs.');
  for (const ref of models) {
    const m = typeof ref === 'string' ? registry.get(ref) : ref;
    if (m.status !== 'stable' && !o.allowPreview) {
      throw new ConfigError(`"${m.id}" is ${m.status}. Pass allowPreview: true to preload it.`);
    }
    await warm(m, { ...o, allowPreview: true });
  }
}

/** Ask the browser not to evict cached model files under storage pressure. Always true on servers. */
export function persist(): Promise<boolean> {
  return getPlatform().persist();
}

/** Storage quota and usage, in bytes. */
export function storage(): Promise<{ quota: number | null; usage: number | null }> {
  return getPlatform().storageEstimate();
}

/** Inspect and clear cached model files. */
export const cache = {
  /** Cached files and their sizes. */
  list(): Promise<{ key: string; bytes: number }[]> {
    return getPlatform().cacheList();
  },
  /** Delete cached files whose key contains `match` (a model's repo name works), or everything. Returns the count. */
  delete(match?: string): Promise<number> {
    if (match && registry.has(match)) {
      const m = registry.get(match);
      const s = m.source;
      match = 'repo' in s ? s.repo : 'bucket' in s ? s.path.split('/')[0] : 'url' in s ? s.url : match;
    }
    return getPlatform().cacheDelete(match);
  },
};
