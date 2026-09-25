import { getConfig } from '../core/config.ts';
import { ConfigError } from '../core/errors.ts';
import { getPlatform } from '../core/runtime.ts';
import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { serialize } from '../core/util.ts';
import { phonemizeText } from './kokoro-text.ts';
import { getTransformers, loadTjs, repoOf } from './transformers.ts';

export interface VoiceInfo {
  id: string;
  name: string;
  lang: 'en-US' | 'en-GB';
  gender: 'female' | 'male';
  grade: string;
}

const V = (id: string, name: string, grade: string): VoiceInfo => ({
  id,
  name,
  lang: id[0] === 'a' ? 'en-US' : 'en-GB',
  gender: id[1] === 'f' ? 'female' : 'male',
  grade,
});

export const KOKORO_VOICES: VoiceInfo[] = [
  V('af_heart', 'Heart', 'A'),
  V('af_bella', 'Bella', 'A-'),
  V('af_nicole', 'Nicole', 'B-'),
  V('af_aoede', 'Aoede', 'C+'),
  V('af_kore', 'Kore', 'C+'),
  V('af_sarah', 'Sarah', 'C+'),
  V('af_alloy', 'Alloy', 'C'),
  V('af_nova', 'Nova', 'C'),
  V('af_sky', 'Sky', 'C-'),
  V('af_jessica', 'Jessica', 'D'),
  V('af_river', 'River', 'D'),
  V('am_fenrir', 'Fenrir', 'C+'),
  V('am_michael', 'Michael', 'C+'),
  V('am_puck', 'Puck', 'C+'),
  V('am_echo', 'Echo', 'D'),
  V('am_eric', 'Eric', 'D'),
  V('am_liam', 'Liam', 'D'),
  V('am_onyx', 'Onyx', 'D'),
  V('am_santa', 'Santa', 'D-'),
  V('am_adam', 'Adam', 'F+'),
  V('bf_emma', 'Emma', 'B-'),
  V('bf_isabella', 'Isabella', 'C'),
  V('bf_alice', 'Alice', 'D'),
  V('bf_lily', 'Lily', 'D'),
  V('bm_george', 'George', 'C'),
  V('bm_fable', 'Fable', 'C'),
  V('bm_lewis', 'Lewis', 'D+'),
  V('bm_daniel', 'Daniel', 'D'),
];

const voiceCache = new Map<string, Promise<Float32Array>>();

async function loadVoice(m: Manifest, id: string, signal?: AbortSignal): Promise<Float32Array> {
  if (!KOKORO_VOICES.some((v) => v.id === id)) {
    throw new ConfigError(`Unknown voice "${id}".`, { hint: `Pick one of: ${KOKORO_VOICES.map((v) => v.id).join(', ')}.` });
  }
  const { repo, revision } = repoOf(m);
  const url = `${getConfig().hub.replace(/\/$/, '')}/${repo}/resolve/${revision}/voices/${id}.bin`;
  let p = voiceCache.get(url);
  if (!p) {
    p = getPlatform()
      .fetchCached(url, { signal })
      .then((b) => new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
    voiceCache.set(url, p);
    p.catch(() => voiceCache.delete(url));
  }
  return p;
}

/** A voice ID, or a weighted blend of voice IDs. */
export type VoiceMix = string | Record<string, number>;

export async function voiceVector(m: Manifest, voice: VoiceMix, signal?: AbortSignal): Promise<{ data: Float32Array; lang: 'a' | 'b' }> {
  if (typeof voice === 'string') return { data: await loadVoice(m, voice, signal), lang: voice[0] === 'b' ? 'b' : 'a' };
  const entries = Object.entries(voice).filter(([, w]) => w > 0);
  if (!entries.length) throw new ConfigError('A voice blend needs at least one voice with a positive weight.');
  const total = entries.reduce((s, [, w]) => s + w, 0);
  const vecs = await Promise.all(entries.map(([id]) => loadVoice(m, id, signal)));
  const out = new Float32Array(vecs[0].length);
  vecs.forEach((v, i) => {
    const w = entries[i][1] / total;
    for (let k = 0; k < out.length; k++) out[k] += v[k] * w;
  });
  const lead = entries.sort((a, b) => b[1] - a[1])[0][0];
  return { data: out, lang: lead[0] === 'b' ? 'b' : 'a' };
}

interface Bundle {
  model: ((i: Record<string, unknown>) => Promise<{ waveform: { data: Float32Array } }>) & { dispose?: () => Promise<void> };
  tokenizer: (t: string, o?: Record<string, unknown>) => { input_ids: { dims: number[]; data: ArrayLike<bigint | number> } };
}

export async function loadKokoro(m: Manifest, opts: CommonOptions) {
  return loadTjs<Bundle>(
    m,
    opts,
    async (t, base) => {
      const { repo } = repoOf(m);
      const [model, tokenizer] = await Promise.all([
        t.StyleTextToSpeech2Model.from_pretrained(repo, base as never),
        t.AutoTokenizer.from_pretrained(repo, { revision: base.revision, progress_callback: base.progress_callback as never }),
      ]);
      return { model: model as unknown as Bundle['model'], tokenizer: tokenizer as unknown as Bundle['tokenizer'] };
    },
    async (b) => {
      await b.model.dispose?.();
    },
  );
}

/** Synthesize one sentence. Returns 24 kHz mono samples. */
export async function kokoroSentence(
  m: Manifest,
  opts: CommonOptions,
  text: string,
  voice: { data: Float32Array; lang: 'a' | 'b' },
  speed: number,
): Promise<{ samples: Float32Array; info: RunInfo }> {
  const t = await getTransformers();
  const loaded = await loadKokoro(m, opts);
  const { model, tokenizer } = loaded.value;
  const phonemes = await phonemizeText(text, voice.lang);
  const run = async (ph: string): Promise<Float32Array> => {
    const { input_ids } = tokenizer(ph, { truncation: true });
    const n = input_ids.dims.at(-1) ?? 0;
    if (n > 510) {
      const words = ph.split(' ');
      const mid = Math.floor(words.length / 2);
      const a = await run(words.slice(0, mid).join(' '));
      const b = await run(words.slice(mid).join(' '));
      const out = new Float32Array(a.length + b.length);
      out.set(a);
      out.set(b, a.length);
      return out;
    }
    const idx = Math.min(Math.max(n - 2, 0), 509);
    const style = voice.data.slice(idx * 256, idx * 256 + 256);
    const { waveform } = await serialize(`tts:${loaded.info.model}:${loaded.info.device}`, () =>
      model({
        input_ids,
        style: new t.Tensor('float32', style, [1, 256]),
        speed: new t.Tensor('float32', [speed], [1]),
      }),
    );
    return waveform.data;
  };
  if (!phonemes) return { samples: new Float32Array(0), info: loaded.info };
  return { samples: await run(phonemes), info: loaded.info };
}
