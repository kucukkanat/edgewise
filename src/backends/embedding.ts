import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { serialize } from '../core/util.ts';
import { loadTjs, repoOf } from './transformers.ts';

interface Bundle {
  tokenizer: (texts: string[], o: Record<string, unknown>) => Record<string, unknown>;
  model: ((inputs: Record<string, unknown>) => Promise<Record<string, { data: Float32Array; dims: number[] }>>) & {
    dispose?: () => Promise<void>;
  };
}

function normalize(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

export async function loadEmbedder(m: Manifest, opts: CommonOptions) {
  return loadTjs<Bundle>(
    m,
    opts,
    async (t, base) => {
      const { repo } = repoOf(m);
      const [tokenizer, model] = await Promise.all([
        t.AutoTokenizer.from_pretrained(repo, { revision: base.revision, progress_callback: base.progress_callback as never }),
        t.AutoModel.from_pretrained(repo, base as never),
      ]);
      return { tokenizer: tokenizer as unknown as Bundle['tokenizer'], model: model as unknown as Bundle['model'] };
    },
    async (b) => {
      await b.model.dispose?.();
    },
  );
}

/** Embed a batch of texts. Returns one Float32Array per text. */
export async function embedTexts(
  m: Manifest,
  opts: CommonOptions,
  texts: string[],
  o: { dimensions?: number; normalize?: boolean; onBatch?: (done: number, total: number) => void },
): Promise<{ vectors: Float32Array[]; info: RunInfo }> {
  const loaded = await loadEmbedder(m, opts);
  const { tokenizer, model } = loaded.value;
  const pooling = (m.config?.pooling as string | undefined) ?? 'mean';
  const vectors: Float32Array[] = [];
  const batch = 16;
  for (let i = 0; i < texts.length; i += batch) {
    const chunk = texts.slice(i, i + batch);
    const out = await serialize(`embed:${loaded.info.model}:${loaded.info.device}`, async () => {
      const inputs = tokenizer(chunk, { padding: true, truncation: true, max_length: (m.config?.maxLength as number | undefined) ?? 512 });
      const res = await model(inputs);
      if (pooling === 'sentence_embedding' && res.sentence_embedding) {
        const { data, dims } = res.sentence_embedding;
        const d = dims[1];
        return chunk.map((_, j) => data.slice(j * d, (j + 1) * d));
      }
      const hidden = res.last_hidden_state ?? res.token_embeddings ?? Object.values(res)[0];
      const [b, seq, d] = hidden.dims;
      const mask = inputs.attention_mask as { data: ArrayLike<number | bigint> };
      const rows: Float32Array[] = [];
      for (let j = 0; j < b; j++) {
        const v = new Float32Array(d);
        if (pooling === 'cls') v.set(hidden.data.subarray(j * seq * d, j * seq * d + d));
        else {
          let n = 0;
          for (let s = 0; s < seq; s++) {
            if (Number(mask.data[j * seq + s]) === 0) continue;
            n++;
            const off = (j * seq + s) * d;
            for (let k = 0; k < d; k++) v[k] += hidden.data[off + k];
          }
          if (n) for (let k = 0; k < d; k++) v[k] /= n;
        }
        rows.push(v);
      }
      return rows;
    });
    for (const v of out) {
      const cut = o.dimensions && o.dimensions < v.length ? v.slice(0, o.dimensions) : v;
      vectors.push(o.normalize === false ? cut : normalize(cut));
    }
    o.onBatch?.(Math.min(texts.length, i + batch), texts.length);
  }
  return { vectors, info: loaded.info };
}
