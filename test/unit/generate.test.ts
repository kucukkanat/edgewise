import { z } from 'zod';
import { AbortError, ConfigError, SchemaValidationError, UnsupportedInputError } from '../../src/core/errors.ts';
import { mockModel } from '../../src/test/index.ts';
import { generate } from '../../src/verbs/generate.ts';
import { tool } from '../../src/verbs/tools.ts';

describe('generate (mock model)', () => {
  it('streams deltas and returns the full result', async () => {
    const model = mockModel({ verb: 'generate', respond: () => 'Hello from the edge' });
    const run = generate({ model, input: 'hi' });
    const deltas: string[] = [];
    for await (const d of run) deltas.push(d as string);
    const r = await run;
    expect(deltas.join('')).toBe('Hello from the edge');
    expect(deltas.length).toBeGreaterThan(1);
    expect(r.text).toBe('Hello from the edge');
    expect(r.finishReason).toBe('stop');
    expect(r.messages.at(-1)).toEqual({ role: 'assistant', content: 'Hello from the edge' });
    expect(r.info.backend).toBe('mock');
  });

  it('passes the conversation and system prompt to the model', async () => {
    let seen: unknown;
    const model = mockModel({ verb: 'generate', respond: ({ messages }) => ((seen = messages), 'ok') });
    await generate({
      model,
      system: 'Be terse.',
      messages: [
        { role: 'user', content: 'a' },
        { role: 'assistant', content: 'b' },
        { role: 'user', content: 'c' },
      ],
    });
    expect(seen).toEqual([
      { role: 'system', content: 'Be terse.' },
      { role: 'user', content: 'a' },
      { role: 'assistant', content: 'b' },
      { role: 'user', content: 'c' },
    ]);
  });

  it('runs tools, feeds results back, and answers', async () => {
    let step = 0;
    const model = mockModel({
      verb: 'generate',
      respond: ({ messages }) => {
        step++;
        if (step === 1) return { toolCalls: [{ name: 'setVolume', input: { level: 30 } }] };
        const last = messages.at(-1);
        return `Done: ${typeof last?.content === 'string' ? last.content : ''}`;
      },
    });
    let volume = 0;
    const run = generate({
      model,
      input: 'Set the volume to 30',
      tools: {
        setVolume: tool({ description: 'Set volume', input: z.object({ level: z.number() }), execute: ({ level }) => ((volume = level), { ok: true }) }),
      },
    });
    const events: string[] = [];
    for await (const e of run.events) events.push(e.type);
    const r = await run;
    expect(volume).toBe(30);
    expect(r.toolCalls).toEqual([{ id: 'call_1', name: 'setVolume', input: { level: 30 } }]);
    expect(r.toolResults).toEqual([{ id: 'call_1', name: 'setVolume', output: { ok: true } }]);
    expect(r.text).toBe('Done: {"ok":true}');
    expect(events).toContain('tool-call');
    expect(events).toContain('tool-result');
  });

  it('returns calls without running them when tools have no execute', async () => {
    const model = mockModel({ verb: 'generate', respond: () => ({ toolCalls: [{ name: 'click', input: { x: 1, y: 2 } }] }) });
    const r = await generate({ model, input: 'click', tools: { click: tool({ description: 'Click', input: z.object({ x: z.number(), y: z.number() }) }) } });
    expect(r.finishReason).toBe('tool-calls');
    expect(r.toolCalls[0].input).toEqual({ x: 1, y: 2 });
    expect(r.toolResults).toEqual([]);
  });

  it('lets approve() decline a call', async () => {
    let step = 0;
    const model = mockModel({
      verb: 'generate',
      respond: ({ messages }) => (++step === 1 ? { toolCalls: [{ name: 'del', input: {} }] } : String(messages.at(-1)?.content)),
    });
    let ran = false;
    const r = await generate({
      model,
      input: 'delete it',
      tools: { del: tool({ description: 'Delete', input: z.object({}), execute: () => ((ran = true), null) }) },
      approve: () => false,
    });
    expect(ran).toBe(false);
    expect(r.toolResults[0].error).toMatch(/declined/);
    expect(r.text).toContain('declined');
  });

  it('reports invalid tool arguments back to the model', async () => {
    let step = 0;
    const model = mockModel({
      verb: 'generate',
      respond: ({ messages }) => (++step === 1 ? { toolCalls: [{ name: 'v', input: { level: 'loud' } }] } : String(messages.at(-1)?.content)),
    });
    const r = await generate({ model, input: 'x', tools: { v: tool({ description: 'v', input: z.object({ level: z.number() }), execute: () => 1 }) } });
    expect(r.toolResults[0].error).toBeTruthy();
  });

  it('stops after maxSteps', async () => {
    const model = mockModel({ verb: 'generate', respond: () => ({ toolCalls: [{ name: 'loop', input: {} }] }) });
    let calls = 0;
    const r = await generate({ model, input: 'x', maxSteps: 2, tools: { loop: tool({ description: 'l', input: z.object({}), execute: () => ++calls }) } });
    expect(calls).toBe(2);
    expect(r.finishReason).toBe('tool-calls');
  });

  it('fills a schema, streaming partial objects', async () => {
    const model = mockModel({ verb: 'generate', respond: () => '{"who": "Ana", "date": "2026-10-02"}' });
    const run = generate({ model, input: 'extract', schema: z.object({ who: z.string(), date: z.string() }) });
    const partials: unknown[] = [];
    for await (const p of run) partials.push(p);
    const r = await run;
    expect(r.object).toEqual({ who: 'Ana', date: '2026-10-02' });
    expect(partials.length).toBeGreaterThan(0);
    expect(partials.at(-1)).toEqual({ who: 'Ana', date: '2026-10-02' });
  });

  it('retries once when the output does not match the schema', async () => {
    let n = 0;
    const model = mockModel({ verb: 'generate', respond: () => (++n === 1 ? '{"who": 3}' : '{"who": "Ana"}') });
    const r = await generate({ model, input: 'x', schema: z.object({ who: z.string() }) });
    expect(n).toBe(2);
    expect(r.object).toEqual({ who: 'Ana' });
  });

  it('throws SchemaValidationError with the raw text when it still fails', async () => {
    const model = mockModel({ verb: 'generate', respond: () => 'not json at all' });
    const err = await generate({ model, input: 'x', schema: z.object({ a: z.number() }) }).catch((e) => e);
    expect(err).toBeInstanceOf(SchemaValidationError);
    expect((err as SchemaValidationError).raw).toBe('not json at all');
  });

  it('can be cancelled', async () => {
    const model = mockModel({ verb: 'generate', respond: async () => (await new Promise((r) => setTimeout(r, 50)), 'late') });
    const ac = new AbortController();
    const run = generate({ model, input: 'x', signal: ac.signal });
    ac.abort();
    await expect(Promise.resolve(run)).rejects.toBeInstanceOf(AbortError);
  });

  it('checks inputs and options before running', async () => {
    await expect(
      Promise.resolve(
        generate({ model: 'lfm2.5-350m', input: [{ type: 'image', image: { data: new Uint8Array(3), width: 1, height: 1, channels: 3 } }, 'x'] }),
      ),
    ).rejects.toBeInstanceOf(UnsupportedInputError);
    await expect(Promise.resolve(generate({ model: 'lfm2.5-350m', input: 'x', preset: 'ocr' }))).rejects.toBeInstanceOf(ConfigError);
    await expect(
      Promise.resolve(generate({ model: 'lfm2.5-vl-450m', input: 'x', tools: { a: tool({ description: 'a', input: z.object({}) }) } })),
    ).rejects.toBeInstanceOf(ConfigError);
  });

  it('tries models in order and uses the first that fits', async () => {
    const model = mockModel({ verb: 'generate', respond: () => 'from mock' });
    const r = await generate({ model: ['kokoro-82m', model], input: 'x' });
    expect(r.text).toBe('from mock');
  });
});
