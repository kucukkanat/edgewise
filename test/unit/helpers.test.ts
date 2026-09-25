import { configure, resetConfig } from '../../src/core/config.ts';
import { cpuScores } from '../../src/gpu/vector.ts';
import { cosine, detectAnomalies, lint, redact, replaceSpans, route, vectorIndex } from '../../src/helpers/index.ts';
import { mockModel } from '../../src/test/index.ts';
import { boolean, spans } from '../../src/verbs/evaluate.ts';

const isBrowser = typeof window !== 'undefined';

describe('helpers', () => {
  it('replaces spans right to left', () => {
    const text = 'Mail ana@x.io or call 555-1234';
    const out = replaceSpans(
      text,
      [
        { start: 5, end: 13, type: 'EMAIL' },
        { start: 22, end: 30, type: 'PHONE' },
      ],
      (s) => `[${s.type}]`,
    );
    expect(out).toBe('Mail [EMAIL] or call [PHONE]');
  });

  it('computes cosine similarity', () => {
    expect(cosine(Float32Array.from([1, 0]), Float32Array.from([2, 0]))).toBeCloseTo(1);
    expect(cosine(Float32Array.from([1, 0]), Float32Array.from([0, 1]))).toBeCloseTo(0);
    expect(() => cosine(new Float32Array(2), new Float32Array(3))).toThrow();
  });

  it('routes with a mock judge', async () => {
    const m = mockModel({ verb: 'evaluate', respond: () => ({ lane: { code: 0.7, chat: 0.3 } }) });
    const r = await route('fix my test', { code: 'Programming', chat: 'Small talk' }, { model: m });
    expect(r.route).toBe('code');
    const low = await route('x', { code: 'Programming', chat: 'Small talk' }, { model: m, threshold: 0.9, otherwise: 'chat' });
    expect(low.route).toBe('chat');
    expect(low.fellBack).toBe(true);
  });

  it('lints policies and redacts', async () => {
    const m = mockModel({ verb: 'evaluate', respond: () => ({ injection: 0.95, pii: [{ type: 'EMAIL', start: 5, end: 13, score: 0.99 }] }) });
    const r = await lint('Mail ana@x.io now', { injection: boolean({ model: m, threshold: 0.8 }), pii: spans(['EMAIL'], { model: m, threshold: 0.9 }) });
    expect(r.passed).toBe(false);
    expect(r.findings.map((f) => f.policy).sort()).toEqual(['injection', 'pii']);
    await expect(lint('x', { a: boolean({ model: m }) })).rejects.toThrow(/threshold/);
    expect(await redact('Mail ana@x.io now', { model: m })).toBe('Mail [EMAIL] now');
  });

  it('flags anomalies from rolling forecasts', async () => {
    const m = mockModel({ verb: 'forecast', respond: ({ series, horizon }) => Array(horizon).fill(series[series.length - 1]) });
    const series = Array.from({ length: 40 }, (_, i) => (i === 30 ? 100 : 1));
    const flags = await detectAnomalies(series, { model: m, warmup: 10 });
    // the mock predicts the last value with zero spread: the spike and the drop back are both flagged
    expect(flags.map((f) => f.index)).toEqual([30, 31]);
    expect(flags[0].direction).toBe('high');
  });
});

describe('vector index', () => {
  beforeAll(() => {
    if (!isBrowser) configure({ cacheDir: `${process.env.TMPDIR ?? '/tmp'}/edgewise-test-${Date.now()}` });
  });
  afterAll(() => resetConfig());

  it('scores on the CPU exactly', () => {
    const s = cpuScores(Float32Array.from([1, 0, 0, 1, 0.6, 0.8]), 2, Float32Array.from([1, 0]));
    expect(Array.from(s).map((x) => Number(x.toFixed(2)))).toEqual([1, 0, 0.6]);
  });

  it('adds, searches, updates, removes and finds duplicates', async () => {
    const idx = await vectorIndex<{ title: string }>({ name: `t-${Math.random()}`, model: 'embed:tiny', load: false });
    await idx.add([
      { id: 'a', vector: Float32Array.from([1, 0, 0]), meta: { title: 'A' } },
      { id: 'b', vector: Float32Array.from([0, 1, 0]), meta: { title: 'B' } },
      { id: 'c', vector: Float32Array.from([0.9, 0.1, 0]), meta: { title: 'C' } },
    ]);
    expect(idx.size).toBe(3);
    const hits = await idx.search(Float32Array.from([1, 0, 0]), { k: 2 });
    expect(hits.map((h) => h.id)).toEqual(['a', 'c']);
    expect(hits[0].meta?.title).toBe('A');
    expect((await idx.search(Float32Array.from([1, 0, 0]), { minScore: 0.999 })).map((h) => h.id)).toEqual(['a']);
    await idx.add({ id: 'b', vector: Float32Array.from([1, 0, 0]) });
    expect(idx.size).toBe(3);
    const pairs = await idx.pairs({ minScore: 0.99 });
    expect(pairs.map((p) => [p.a, p.b].sort().join())).toContain('a,b');
    expect(await idx.remove('a')).toBe(true);
    expect(await idx.remove('zzz')).toBe(false);
    expect(idx.size).toBe(2);
    await expect(idx.add({ id: 'x', vector: new Float32Array(5) })).rejects.toThrow(/dimensions/);
  });

  it('saves and loads', async () => {
    const name = `persist-${Math.random().toString(36).slice(2)}`;
    const a = await vectorIndex({ name, model: 'embed:tiny' });
    await a.add([{ id: 'x', vector: Float32Array.from([0, 1]), meta: { n: 1 } }]);
    await a.save();
    const b = await vectorIndex<{ n: number }>({ name, model: 'embed:tiny' });
    expect(b.size).toBe(1);
    expect((await b.search(Float32Array.from([0, 1])))[0]).toMatchObject({ id: 'x', meta: { n: 1 } });
    const other = await vectorIndex({ name, model: 'embed:default' });
    expect(other.size).toBe(0);
  });
});
