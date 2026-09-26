import { z } from 'zod';
import { AbortError, choice, SchemaValidationError, tool } from '../../src/index.ts';
import { connectWorker, type EdgewiseWorker, serveWorker } from '../../src/worker/index.ts';
import { defineWorkerMocks } from '../fixtures/mocks.ts';

function inProcess(): EdgewiseWorker {
  const { port1, port2 } = new MessageChannel();
  serveWorker(port1 as never);
  return connectWorker(port2 as never);
}

async function suiteFor(make: () => EdgewiseWorker | Promise<EdgewiseWorker>) {
  let ew: EdgewiseWorker;
  beforeAll(async () => {
    ew = await make();
  });

  it('streams generate and returns the result', async () => {
    const run = ew.generate({ model: 'wk:echo', input: 'hello' });
    let text = '';
    for await (const d of run) text += d;
    const r = await run;
    expect(r.text).toBe('echo: hello');
    expect(text).toBe('echo: hello');
  });

  it('runs tools on the calling side', async () => {
    const seen: unknown[] = [];
    const r = await ew.generate({
      model: 'wk:tools',
      input: 'add',
      tools: {
        add: tool({
          description: 'Add two numbers',
          input: z.object({ a: z.number(), b: z.number() }),
          execute: async ({ a, b }) => {
            seen.push([a, b]);
            return { sum: a + b };
          },
        }),
      },
    });
    expect(seen).toEqual([[2, 3]]);
    expect(r.toolResults[0].output).toEqual({ sum: 5 });
    expect(r.text).toContain('"sum":5');
  });

  it('checks schemas on the calling side', async () => {
    await expect(ew.generate({ model: 'wk:json', input: 'x', schema: z.object({ name: z.string(), age: z.number() }) }).then((r) => r)).rejects.toBeInstanceOf(
      SchemaValidationError,
    );
  });

  it('evaluates, embeds and forecasts', async () => {
    const e = await ew.evaluate({ model: 'wk:judge', state: 'x', questions: { lane: choice({ billing: 'billing', other: 'other' }) } });
    expect(e.answers.lane.choice).toBe('billing');
    const m = await ew.embed({ model: 'wk:embed', values: ['a', 'abc'] });
    expect(m.embeddings).toHaveLength(2);
    expect(m.embeddings[1]).toBeInstanceOf(Float32Array);
    const f = await ew.forecast({ model: 'wk:fc', series: [1, 2, 3, 4], horizon: 3 });
    expect(Array.from(f.median)).toEqual([0, 1, 2]);
  });

  it('speaks a streamed input and rebuilds SpeechAudio', async () => {
    const deltas = (async function* () {
      yield 'One. ';
      yield 'Two.';
    })();
    const run = ew.speak({ model: 'wk:voice', input: deltas });
    const texts: string[] = [];
    for await (const c of run) texts.push(c.text);
    const audio = await run;
    expect(texts).toEqual(['One.', 'Two.']);
    expect(audio.toWav().length).toBeGreaterThan(44);
  });

  it('cancels a run', async () => {
    const run = ew.generate({ model: 'wk:slow', input: 'x' });
    setTimeout(() => run.cancel(), 20);
    await expect(run.then((r) => r)).rejects.toBeInstanceOf(AbortError);
  });

  it('reports worker errors with their code and hint', async () => {
    const err = (await ew.generate({ model: 'no-such-model', input: 'x' }).then(
      () => null,
      (e) => e,
    )) as { code?: string; name?: string; message?: string };
    expect(err.code).toBe('E_MODEL');
    expect(err.name).toBe('ModelNotFoundError');
  });

  it('lists models registered in the worker', async () => {
    expect(await ew.models()).toContain('wk:echo');
  });
}

describe('worker bridge (MessageChannel)', () => {
  defineWorkerMocks();
  suiteFor(inProcess);
});

const hasWorker = typeof Worker !== 'undefined';
(hasWorker ? describe : describe.skip)('worker bridge (real Worker)', () => {
  suiteFor(() => connectWorker(new Worker(new URL('../fixtures/edgewise.worker.ts', import.meta.url), { type: 'module' })));
});
