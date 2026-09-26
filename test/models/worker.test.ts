import { cosine } from '../../src/helpers/index.ts';
import { connectWorker } from '../../src/worker/index.ts';
import { it2, on, report, suite, testConfig } from './setup.ts';

suite('worker mode · real models', () => {
  it2(on(['bun', 'browser'], 'embeds and generates inside a Web Worker'), async () => {
    const w = new Worker(new URL('../fixtures/real.worker.ts', import.meta.url), { type: 'module' });
    const ew = connectWorker(w);
    try {
      await ew.configure(testConfig);
      const { embeddings, info } = await ew.embed({
        model: 'all-minilm-l6-v2',
        values: ['How do I reset my password?', 'I forgot my login', 'Pizza in Naples'],
      });
      report('worker embed', info);
      expect(cosine(embeddings[0], embeddings[1])).toBeGreaterThan(cosine(embeddings[0], embeddings[2]));
      const loads: string[] = [];
      const run = ew.generate({ model: 'text:tiny', input: 'What is the capital of France?', maxTokens: 24, onProgress: (e) => loads.push(e.type) });
      let text = '';
      for await (const d of run) text += d;
      expect(text).toMatch(/Paris/i);
      expect(loads).toContain('ready');
    } finally {
      ew.terminate();
    }
  });
});
