import { aliases, imageMarkup, optionCodes, questionBlock, readoutIds, renderPrompt, stateBlock } from '../../src/backends/d1causal.ts';
import { ConfigError } from '../../src/core/errors.ts';

// A tokenizer where every string up to three characters is one token, and anything longer is one per character.
const tokenize = (s: string) =>
  s.length <= 3 ? [s.split('').reduce((h, c) => h * 131 + (c.codePointAt(0) ?? 0), 7)] : Array.from(s, (c) => c.codePointAt(0) ?? 0);
const one = (s: string) => tokenize(s)[0];

describe('d1-3B prompt', () => {
  it('writes text as itself and other states as indented JSON', () => {
    expect(stateBlock('hello')).toBe('hello\n\n');
    expect(stateBlock({ a: 1, b: ['x'] })).toBe('{\n  "a": 1,\n  "b": [\n    "x"\n  ]\n}\n\n');
  });

  it('codes options with letters, or digit pairs past 26', () => {
    expect(optionCodes(['billing', 'tech'])).toEqual(['A', 'B']);
    expect(optionCodes(['a', 'b', 'c'])).toEqual(['a', 'b', 'c']);
    expect(optionCodes(Array.from({ length: 30 }, (_, i) => `o${i}`)).slice(0, 3)).toEqual(['00', '01', '02']);
  });

  it('gives every option a distinct single token, falling back when a code is not one', () => {
    const codes = aliases(tokenize, ['x', 'y']);
    expect(codes.map(([c]) => c)).toEqual(['x', 'y']);
    expect(new Set(codes.map(([, t]) => t)).size).toBe(2);
    // A tokenizer where "A" is two tokens: the first free fallback stands in.
    const odd = (s: string) => (s === 'A' ? [1, 2] : tokenize(s));
    expect(aliases(odd, ['billing', 'tech']).map(([c]) => c)).toEqual(['B', 'C']);
  });

  it('writes each question type as d1-3B was trained', () => {
    const choice = {
      type: 'choice',
      instructions: 'Which team?',
      options: [
        ['billing', 'payments'],
        ['tech_support', ''],
      ],
    } as const;
    expect(questionBlock(tokenize, choice)).toBe('Which team?\n\nOptions:\nA payments\nB tech support\n\nReply with the option code only.');
    expect(questionBlock(tokenize, { type: 'noul', instructions: 'Refund?' })).toBe('Refund?\n\nReply with yes or no only.');
    expect(questionBlock(tokenize, { type: 'noul', instructions: 'Refund?', true: 'asks for money' })).toBe(
      'Refund?\nYes: asks for money\nNo: None\n\nReply with yes or no only.',
    );
    expect(questionBlock(tokenize, { type: 'score', instructions: 'How urgent?', levels: ['low', 'high'] })).toBe(
      'How urgent?\n\n0 low\n1 high\n\nReply with a single digit 0-1 only.',
    );
  });

  it('reads yes/no, digits and option codes', () => {
    expect(readoutIds(tokenize, { type: 'noul', instructions: 'q' })).toEqual([
      [one('yes'), one('Yes'), one('YES')],
      [one('no'), one('No'), one('NO')],
    ]);
    expect(readoutIds(tokenize, { type: 'score', instructions: 'q', levels: ['a', 'b', 'c'] })).toEqual([[one('0')], [one('1')], [one('2')]]);
    expect(
      readoutIds(tokenize, {
        type: 'choice',
        instructions: 'q',
        options: [
          ['p', ''],
          ['q', ''],
        ],
      }),
    ).toEqual([
      [one('p'), one(' p')],
      [one('q'), one(' q')],
    ]);
  });

  it('fails when an option has no single token to read', () => {
    const wide = (s: string) => (s === '1' || s.toLowerCase() === 'no' ? [1, 1] : tokenize(s));
    expect(() => readoutIds(wide, { type: 'score', instructions: 'q', levels: ['a', 'b'] })).toThrow(ConfigError);
    expect(() => readoutIds(wide, { type: 'noul', instructions: 'q' })).toThrow(ConfigError);
    // No code left: every fallback is taken or more than one token.
    expect(() => aliases(() => [1, 2], ['a', 'b'])).toThrow(ConfigError);
  });

  it('assembles the chat prompt, with or without a state', () => {
    const q = { type: 'noul', instructions: 'Refund?' } as const;
    expect(renderPrompt(tokenize, '<s>', 'I want my money', q)).toBe(
      '<s><|im_start|>user\nI want my money\n\n\nQUESTION:\nRefund?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n',
    );
    expect(renderPrompt(tokenize, '', null, q, '<IMG>')).toBe(
      '<|im_start|>user\n<IMG>Refund?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n',
    );
  });

  it("expands an image into LFM2-VL's tile and thumbnail markup", () => {
    expect(imageMarkup(1, 1, 3)).toBe('<|image_start|><image><image><image><|image_end|>');
    const tiled = imageMarkup(1, 2, 2, 1);
    expect(tiled).toBe('<|image_start|><|img_row_1_col_1|><image><|img_row_1_col_2|><image><|img_thumbnail|><image><image><|image_end|>');
  });
});
