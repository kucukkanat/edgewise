import { z } from 'zod';
import { caption } from '../../src/helpers/index.ts';
import { generate, tool } from '../../src/index.ts';
import { it2, on, redDisc, report, suite } from './setup.ts';

const all = ['bun', 'node', 'browser'];

suite('generate · real models', () => {
  it2(on(all, 'streams text from lfm2.5-350m'), async () => {
    const run = generate({ model: 'lfm2.5-350m', input: 'Name three planets of the solar system. Answer briefly.', maxTokens: 48 });
    let streamed = '';
    for await (const d of run) streamed += d;
    const r = await run;
    report('lfm2.5-350m text', r.info);
    expect(r.text.length).toBeGreaterThan(5);
    expect(streamed.trim()).toBe(r.text.trim());
    expect(r.text).toMatch(/Mercury|Venus|Earth|Mars|Jupiter|Saturn|Uranus|Neptune/i);
    expect(r.usage.outputTokens).toBeGreaterThan(0);
  });

  it2(on(all, 'streams text from lfm2.5-230m (text:tiny)'), async () => {
    const r = await generate({ model: 'text:tiny', input: 'What is the capital of France?', maxTokens: 24 });
    report('lfm2.5-230m', r.info);
    expect(r.text).toMatch(/Paris/i);
  });

  it2(on(['bun', 'node'], 'answers with lfm2.5-1.2b (needs WebGPU in browsers)'), async () => {
    const r = await generate({ model: 'lfm2.5-1.2b', input: 'What is the capital of Italy? Answer in one word.', maxTokens: 12 });
    report('lfm2.5-1.2b', r.info);
    expect(r.text).toMatch(/Rome/i);
  });

  it2(on(all, 'calls a tool and feeds the result back'), async () => {
    let level = -1;
    const r = await generate({
      model: 'lfm2.5-350m',
      input: 'Set the volume to 30.',
      tools: {
        setVolume: tool({
          description: 'Set the speaker volume from 0 to 100',
          input: z.object({ level: z.number().int() }),
          execute: async (a) => {
            level = a.level;
            return { ok: true };
          },
        }),
      },
    });
    expect(r.toolCalls.map((c) => c.name)).toContain('setVolume');
    expect(level).toBe(30);
    expect(r.toolResults.length).toBeGreaterThan(0);
  });

  it2(on(all, 'returns a typed object for a schema'), async () => {
    const r = await generate({
      model: 'lfm2.5-350m',
      input: 'Extract the event: "Lunch with Ana on 2026-10-02 at Foodhallen"',
      schema: z.object({ who: z.string(), date: z.string(), place: z.string().optional() }),
    });
    expect(r.object?.who).toMatch(/Ana/);
    expect(r.object?.date).toMatch(/2026-10-02|October/);
  });

  it2(on(all, 'answers a question about an image (lfm2.5-vl-450m)'), async () => {
    const r = await generate({ model: 'lfm2.5-vl-450m', input: [redDisc(), 'What color is the circle? Answer with one word.'], maxTokens: 16 });
    report('lfm2.5-vl-450m', r.info);
    expect(r.text.toLowerCase()).toContain('red');
  });

  it2(on(all, 'captions an image with Florence-2'), async () => {
    const r = await caption(redDisc(384), { model: 'florence-2-base' });
    report('florence-2-base', r.info);
    expect(r.text.length).toBeGreaterThan(3);
  });
});
