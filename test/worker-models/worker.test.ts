import { cosine } from '../../src/helpers/index.ts';
import { choice } from '../../src/index.ts';
import { connectWorker, type EdgewiseWorker } from '../../src/worker/index.ts';
import { it2, on, redDisc, report, suite, testConfig } from '../models/setup.ts';

suite('worker mode · real models', () => {
  // One worker for the suite, so each model loads once.
  let ew: EdgewiseWorker | undefined;
  const worker = async () => {
    if (!ew) {
      ew = connectWorker(new Worker(new URL('../fixtures/real.worker.ts', import.meta.url), { type: 'module' }));
      await ew.configure(testConfig);
    }
    return ew;
  };
  afterAll(async () => {
    await ew?.terminate();
  });

  it2(on(['bun', 'browser'], 'embeds and generates inside a Web Worker'), async () => {
    const ew = await worker();
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
  });

  it2(on(['bun', 'browser'], 'judges an image with d1-omni inside a Web Worker'), async () => {
    const ew = await worker();
    const { results } = await ew.evaluate({
      model: 'd1-omni-600m',
      items: [{ images: redDisc(256) }],
      questions: { shape: choice({ circle: '', square: '', triangle: '' }, { instructions: 'What shape is shown?' }) },
    });
    report('worker d1', results[0].info.shape);
    expect(results[0].answers.shape.choice).toBe('circle');
  });
});
