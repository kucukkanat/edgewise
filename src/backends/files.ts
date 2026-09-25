import { getConfig } from '../core/config.ts';
import { ConfigError } from '../core/errors.ts';
import { getPlatform } from '../core/runtime.ts';
import type { LoadEvent, Manifest } from '../core/types.ts';

/** URL of a file inside a model's source. */
export function fileUrl(m: Manifest, rel: string): string {
  const s = m.source;
  if ('repo' in s) {
    const hub = getConfig().hub.replace(/\/$/, '');
    const sub = s.subfolder ? `${s.subfolder.replace(/\/$/, '')}/` : '';
    return `${hub}/${s.repo}/resolve/${s.revision}/${sub}${rel}`;
  }
  if ('github' in s) return `${getConfig().githubRaw.replace(/\/$/, '')}/${s.github}/${s.revision}/${s.path.replace(/\/$/, '')}/${rel}`;
  if ('bucket' in s) {
    const hub = getConfig().hub.replace(/\/$/, '');
    return `${hub}/buckets/${s.bucket}/resolve/${s.path.replace(/\/$/, '')}/${rel}`;
  }
  if ('url' in s) return `${s.url.replace(/\/$/, '')}/${rel}`;
  throw new ConfigError(`"${m.id}" has no downloadable files.`);
}

/**
 * Download a model file (optionally split into `.part0 … .partN` pieces) through the cache.
 * Returns a path on servers and bytes in browsers.
 */
export async function fetchModelFile(
  m: Manifest,
  rel: string,
  o: { parts?: number; signal?: AbortSignal; onProgress?: (e: LoadEvent) => void } = {},
): Promise<Uint8Array | string> {
  const p = getPlatform();
  const sha256 = (m.config?.sha256 as Record<string, string> | undefined)?.[rel];
  const progress = (loaded: number, total: number) => o.onProgress?.({ type: 'download', model: m.id, file: rel, loaded, total });
  if (!o.parts || o.parts <= 1) return p.fetchModel(fileUrl(m, rel), { signal: o.signal, onProgress: progress, sha256 });
  const pieces: Uint8Array[] = [];
  let done = 0;
  for (let i = 0; i < o.parts; i++) {
    const b = await p.fetchCached(fileUrl(m, `${rel}.part${i}`), {
      signal: o.signal,
      onProgress: (l) => progress(done + l, 0),
    });
    done += b.byteLength;
    pieces.push(b);
  }
  const out = new Uint8Array(done);
  let off = 0;
  for (const b of pieces) {
    out.set(b, off);
    off += b.byteLength;
  }
  return out;
}
