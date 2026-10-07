import {
  encodeQuestion,
  escapeDelimiters,
  layoutRows,
  packRows,
  pyJson,
  renderOptions,
  resizeWeights,
  subsampledLength,
  temperatureOf,
} from '../../src/backends/d1.ts';
import { ConfigError, UnsupportedInputError } from '../../src/core/errors.ts';
import { mockModel } from '../../src/test/index.ts';
import { boolean, choice, evaluate, label, score } from '../../src/verbs/evaluate.ts';

// One token per character, so positions are easy to read.
const chars = (s: string) => Array.from(s, (c) => 1000 + (c.codePointAt(0) ?? 0));

describe('d1 prompt encoding', () => {
  it('escapes delimiters in caller text', () => {
    expect(escapeDelimiters('a <|mask|> b <|reserved_7|> <|not a token|>')).toBe('a <¦mask¦> b <¦reserved_7¦> <|not a token|>');
  });

  it('serializes JSON the way Python json.dumps does', () => {
    expect(pyJson({ a: 1, b: [true, null, 'é'], c: { d: 'x' }, skip: undefined })).toBe('{"a": 1, "b": [true, null, "é"], "c": {"d": "x"}}');
    expect(pyJson('text')).toBe('"text"');
    expect(pyJson(undefined)).toBe('null');
  });

  it('renders options per type and medium', () => {
    const c = {
      type: 'choice',
      instructions: 'Which?',
      options: [
        ['billing', 'Charges'],
        ['tech', ''],
      ],
    } as const;
    expect(renderOptions(c, 'text')).toEqual(['billing: Charges', 'tech']);
    expect(renderOptions(c, 'audio')).toEqual(['option_000: Charges', 'option_001: tech']);
    expect(renderOptions({ type: 'score', instructions: 'How?', levels: ['low', 'high'] }, 'text')).toEqual(['level 0: low', 'level 1: high']);
    const n = { type: 'noul', instructions: 'Refund?' } as const;
    expect(renderOptions(n, 'text')).toEqual(['false: no, the statement does not hold', 'true: yes, the statement holds']);
    expect(renderOptions(n, 'image')).toEqual(['false: no', 'true: yes']);
    expect(renderOptions({ ...n, true: 'asks for money' }, 'image')).toEqual(['false: no, the statement does not hold', 'true: asks for money']);
    expect(renderOptions({ ...n, true: 'asks for money' }, 'audio')).toEqual(['false: no', 'true: yes']);
  });

  it('lays out state, question and option markers', () => {
    const q = { type: 'noul', instructions: 'Q?', true: 'y', false: 'n' } as const;
    const { ids, markers } = encodeQuestion(chars, 'hi', q, 512, 'text');
    // <bos> <state> h i <q> Q ? <opt> <mask> " false: n" </opt> <opt> <mask> " true: y" </opt> <decide>
    expect(ids.slice(0, 7)).toEqual([1, 17, ...chars('hi'), 18, ...chars('Q?')]);
    expect(markers).toEqual([8, 8 + 1 + ' false: n'.length + 2]);
    for (const m of markers) expect(ids[m]).toBe(16);
    expect(ids[markers[0] - 1]).toBe(19);
    expect(ids.at(-1)).toBe(21);
  });

  it('cuts the state, never the options', () => {
    const q = {
      type: 'choice',
      instructions: 'Which?',
      options: [
        ['a', 'first'],
        ['b', 'second'],
      ],
    } as const;
    const { ids, markers } = encodeQuestion(chars, 'x'.repeat(5000), q, 200, 'text');
    expect(ids.length).toBeLessThanOrEqual(200);
    expect(ids.at(-1)).toBe(21);
    for (const m of markers) expect(ids[m]).toBe(16);
  });

  it('caps each option at its share of the budget', () => {
    const q = {
      type: 'choice',
      instructions: 'Which?',
      options: [
        ['a', 'y'.repeat(500)],
        ['b', 'z'],
      ],
    } as const;
    const { ids, markers } = encodeQuestion(chars, '', q, 16384, 'text');
    // budget = max(96, min(2·24 + 32, 8192)) = 96; per option = (96 − 6) / 2 = 45 tokens.
    expect(markers[1] - markers[0]).toBe(1 + 45 + 2);
    expect(ids.length).toBeLessThan(120);
  });

  it('fails when the options cannot fit', () => {
    const options = Array.from({ length: 40 }, (_, i) => [`o${i}`, 'w'] as const);
    expect(() => encodeQuestion(chars, '', { type: 'choice', instructions: 'Which?', options }, 64, 'text')).toThrow(ConfigError);
  });

  it('reports what it had to cut', () => {
    const q = {
      type: 'choice',
      instructions: 'Which?',
      options: [
        ['a', 'first'],
        ['b', 'second'],
      ],
    } as const;
    expect(encodeQuestion(chars, 'short', q, 16384, 'text').truncated).toBe(false);
    expect(encodeQuestion(chars, 'x'.repeat(500), q, 200, 'text').truncated).toBe(true);
    const longOption = {
      type: 'choice',
      instructions: 'Which?',
      options: [
        ['a', 'y'.repeat(500)],
        ['b', 'z'],
      ],
    } as const;
    expect(encodeQuestion(chars, '', longOption, 16384, 'text').truncated).toBe(true);
    const longInstructions = { type: 'noul', instructions: 'w'.repeat(500) } as const;
    expect(encodeQuestion(chars, '', longInstructions, 16384, 'text').truncated).toBe(true);
  });

  it('packs rows into runs under the token budget, in order', () => {
    const rows = [10, 10, 10, 30, 5];
    const runs = packRows(rows, (r) => r, 40);
    expect(runs).toEqual([[10, 10, 10], [30], [5]]);
    expect(runs.flat()).toEqual(rows);
    // A row longer than the budget still runs, alone.
    expect(packRows([50, 1], (r) => r, 40)).toEqual([[50], [1]]);
    expect(packRows([], (r: number) => r)).toEqual([]);
  });

  it('lays rows out with each prefix left-padded against its text', () => {
    const emb = (n: number, v: number) => new Float32Array(n * 1024).fill(v);
    const rows = [
      { prefix: null, ids: [1, 2, 3], markers: [1] },
      { prefix: emb(2, 7), ids: [4, 5], markers: [0, 1] },
      { prefix: emb(3, 9), ids: [6], markers: [0] },
    ];
    const l = layoutRows(rows);
    expect([l.B, l.T, l.K, l.P]).toEqual([3, 3, 2, 3]);
    expect(Array.from(l.ids, Number)).toEqual([1, 2, 3, 4, 5, 0, 6, 0, 0]);
    expect(Array.from(l.mask, Number)).toEqual([1, 1, 1, 1, 1, 0, 1, 0, 0]);
    expect(Array.from(l.markerPos, Number)).toEqual([1, 0, 0, 1, 0, 0]);
    expect(Array.from(l.markerMask, Number)).toEqual([1, 0, 1, 1, 1, 0]);
    // Text-only: fully masked. Two media rows: padding first, so the media end right before the text.
    expect(Array.from(l.prefixMask, Number)).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 1]);
    const at = (b: number, p: number) => l.prefix[(b * l.P + p) * 1024];
    expect([at(0, 0), at(0, 2), at(1, 0), at(1, 1), at(1, 2), at(2, 0), at(2, 2)]).toEqual([0, 0, 0, 7, 7, 9, 9]);
  });

  it('gives a text-only batch one masked prefix position', () => {
    const l = layoutRows([{ prefix: null, ids: [1], markers: [0] }]);
    expect(l.P).toBe(1);
    expect(Array.from(l.prefixMask, Number)).toEqual([0]);
  });

  it("matches PyTorch's antialiased bilinear resize weights", () => {
    // F.interpolate(eye(16), mode='bilinear', antialias=True), first row; downscaling spreads a row over many inputs.
    const two = [0.080357, 0.098214, 0.116071, 0.133929, 0.133929, 0.116071, 0.098214, 0.080357, 0.0625, 0.044643, 0.026786, 0.008929, 0, 0, 0, 0];
    const w2 = resizeWeights(2);
    two.forEach((v, j) => expect(w2[j]).toBeCloseTo(v, 5));
    // The second row mirrors the first.
    for (let j = 0; j < 16; j++) expect(w2[16 + j]).toBeCloseTo(w2[15 - j], 6);
    const w24 = resizeWeights(24);
    expect(Array.from(w24.subarray(0, 16))).toEqual([1, ...Array(15).fill(0)]);
    for (const n of [2, 5, 16, 24, 64]) {
      const w = resizeWeights(n);
      for (let i = 0; i < n; i++) expect(w.subarray(i * 16, i * 16 + 16).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    }
    // Same size: the identity.
    const id = resizeWeights(16);
    for (let i = 0; i < 16; i++) expect(id[i * 16 + i]).toBeCloseTo(1, 6);
  });

  it('picks the learned temperature by type and option count', () => {
    const temps = { 'choice:2': 1.7, 'choice:3-5': 1.4, 'choice:6-10': 1.2, 'choice:11+': 1.3, 'noul:2': 1.6, score: 1 };
    expect(temperatureOf(temps, 'choice', 2)).toBe(1.7);
    expect(temperatureOf(temps, 'choice', 5)).toBe(1.4);
    expect(temperatureOf(temps, 'choice', 10)).toBe(1.2);
    expect(temperatureOf(temps, 'choice', 40)).toBe(1.3);
    expect(temperatureOf(temps, 'noul', 2)).toBe(1.6);
    expect(temperatureOf(temps, 'score', 4)).toBe(1);
    expect(temperatureOf({}, 'score', 4)).toBe(1);
  });

  it('counts conformer positions after 8x subsampling', () => {
    expect(subsampledLength(50)).toBe(7);
    expect(subsampledLength(3000)).toBe(375);
  });
});

describe('evaluate with media', () => {
  const judge = mockModel({ verb: 'evaluate', respond: () => ({ x: { a: 1, b: 0 } }) });
  const px = { data: new Uint8Array(12), width: 2, height: 2, channels: 3 as const };

  it('needs a state, images or audio', async () => {
    await expect(evaluate({ model: judge, questions: { x: choice({ a: 'A', b: 'B' }) } })).rejects.toThrow(ConfigError);
  });

  it('rejects images on a model that does not accept them', async () => {
    await expect(evaluate({ model: 'judge:router', state: 's', images: px, questions: { x: choice({ a: 'A', b: 'B' }) } })).rejects.toThrow(
      UnsupportedInputError,
    );
  });

  it('checks d1 questions before loading the model', async () => {
    const q = (questions: Record<string, ReturnType<typeof boolean> | ReturnType<typeof label>>) => evaluate({ model: 'judge:omni', state: 's', questions });
    await expect(q({ refund: boolean() })).rejects.toThrow(ConfigError);
    await expect(q({ kind: label() })).rejects.toThrow(UnsupportedInputError);
  });

  it('judges many items with the same questions', async () => {
    const refunds = mockModel({ verb: 'evaluate', respond: ({ state }) => ({ refund: state.includes('money') ? 0.9 : 0.1 }) });
    const { results } = await evaluate({
      model: refunds,
      items: [{ state: 'I want my money back' }, { state: 'Nice app' }, { images: px }],
      questions: { refund: boolean({ true: 'asks for a refund' }) },
    });
    expect(results.map((r) => r.answers.refund.probability)).toEqual([0.9, 0.1, 0.1]);
    expect((await evaluate({ model: refunds, items: [], questions: { refund: boolean({ true: 'refund' }) } })).results).toEqual([]);
  });

  it('keeps items and a top-level state apart', async () => {
    const opts = { model: judge, items: [{ state: 'a' }], state: 'b', questions: { x: choice({ a: 'A', b: 'B' }) } };
    await expect(evaluate(opts as never)).rejects.toThrow(ConfigError);
    await expect(evaluate({ model: judge, items: [{}], questions: { x: choice({ a: 'A', b: 'B' }) } })).rejects.toThrow(ConfigError);
  });

  it('resolves judge:omni to a model that takes text, images and audio', async () => {
    const { registry } = await import('../../src/index.ts');
    const m = registry.get('judge:omni');
    expect(m?.accepts).toEqual(['text', 'image', 'audio']);
    expect(m?.features).toEqual(['choice', 'score', 'boolean']);
    expect(score(['a', 'b']).kind).toBe('score');
  });
});
