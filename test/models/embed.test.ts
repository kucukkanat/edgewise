import { cosine, vectorIndex } from '../../src/helpers/index.ts';
import { embed } from '../../src/index.ts';
import { it2, on, report, suite } from './setup.ts';

const all = ['bun', 'node', 'browser'];

for (const model of ['all-minilm-l6-v2', 'embeddinggemma-300m']) {
  suite(`embed · ${model}`, () => {
    it2(on(all, 'places similar meanings closer'), async () => {
      const { embeddings, dimensions, info } = await embed({
        model,
        values: ['How do I reset my password?', 'I forgot my login credentials', 'Best pizza in Naples'],
      });
      report(model, info);
      expect(embeddings).toHaveLength(3);
      expect(dimensions).toBeGreaterThan(100);
      expect(cosine(embeddings[0], embeddings[1])).toBeGreaterThan(cosine(embeddings[0], embeddings[2]));
      const norm = Math.hypot(...embeddings[0]);
      expect(norm).toBeCloseTo(1, 3);
    });
  });
}

suite('embed · extras', () => {
  it2(on(all, 'shortens EmbeddingGemma vectors (Matryoshka)'), async () => {
    const r = await embed({ model: 'embeddinggemma-300m', input: 'hello', dimensions: 128, purpose: 'query' });
    expect(r.embedding).toHaveLength(128);
    expect(r.model).toBe('embeddinggemma-300m@128');
  });

  it2(on(all, 'searches a vector index'), async () => {
    const index = await vectorIndex<{ id: number }>({ name: 'model-test', model: 'all-minilm-l6-v2', load: false });
    const docs = ['Cats are small furry pets', 'The stock market fell today', 'Dogs love to play fetch'];
    const { embeddings } = await embed({ model: 'all-minilm-l6-v2', values: docs, purpose: 'document' });
    await index.add(embeddings.map((vector, i) => ({ id: `d${i}`, vector, meta: { id: i + 1 } })));
    const hits = await index.query('kittens', { k: 1 });
    expect(hits[0].meta?.id).toBe(1);
  });
});

suite('embed · explicit WebGPU', () => {
  it2(on(['browser'], 'runs on WebGPU when asked, even on a software adapter'), async () => {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    if (!gpu || !(await gpu.requestAdapter())) return;
    const r = await embed({ model: 'all-minilm-l6-v2', input: 'hello', device: 'webgpu' });
    report('minilm webgpu', r.info);
    expect(r.info.device).toBe('webgpu');
    expect(r.embedding).toHaveLength(384);
  });
});
