import { encodeQuestion, escapeDelimiters, pyJson, renderOptions, subsampledLength, temperatureOf } from '../../src/backends/d1.ts';
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
    const q = (questions: Record<string, ReturnType<typeof boolean> | ReturnType<typeof label>>) =>
      evaluate({ model: 'judge:omni', state: 's', allowPreview: true, questions });
    await expect(q({ refund: boolean() })).rejects.toThrow(ConfigError);
    await expect(q({ kind: label() })).rejects.toThrow(UnsupportedInputError);
  });

  it('resolves judge:omni to a model that takes text, images and audio', async () => {
    const { registry } = await import('../../src/index.ts');
    const m = registry.get('judge:omni');
    expect(m?.accepts).toEqual(['text', 'image', 'audio']);
    expect(m?.features).toEqual(['choice', 'score', 'boolean']);
    expect(score(['a', 'b']).kind).toBe('score');
  });
});
