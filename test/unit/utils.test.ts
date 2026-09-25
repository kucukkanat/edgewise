import { alignTokens } from '../../src/backends/judge.ts';
import { decodeOffsets } from '../../src/backends/lfm.ts';
import { stripSpecial, visibleTextFilter } from '../../src/backends/lm.ts';
import { gaussian, latentPreview, sigmaAt } from '../../src/backends/sdturbo.ts';
import { confidenceOf, serialize, softmax } from '../../src/core/util.ts';

describe('numeric helpers', () => {
  it('softmax sums to one and is stable for large logits', () => {
    const p = softmax([1000, 1001, 1002]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p[2]).toBeGreaterThan(p[1]);
  });
  it('confidence is 1 minus normalized entropy', () => {
    expect(confidenceOf([1, 0, 0])).toBeCloseTo(1);
    expect(confidenceOf([1 / 3, 1 / 3, 1 / 3])).toBeCloseTo(0);
    expect(confidenceOf([1])).toBe(1);
  });
  it('serializes work per key', async () => {
    const order: string[] = [];
    const slow = (id: string, ms: number) =>
      serialize('k', async () => {
        order.push(`start ${id}`);
        await new Promise((r) => setTimeout(r, ms));
        order.push(`end ${id}`);
      });
    await Promise.all([slow('a', 20), slow('b', 1)]);
    expect(order).toEqual(['start a', 'end a', 'start b', 'end b']);
  });
});

describe('SD-Turbo sampling', () => {
  it('uses the SD 2.x noise schedule', () => {
    expect(sigmaAt(999)).toBeCloseTo(14.6146, 3);
    expect(sigmaAt(0)).toBeCloseTo(0.0292, 3);
  });
  it('is deterministic for a seed', () => {
    const a = gaussian(42);
    const b = gaussian(42);
    const xs = Array.from({ length: 5 }, () => a());
    expect(Array.from({ length: 5 }, () => b())).toEqual(xs);
    const c = gaussian(43);
    expect(c()).not.toBe(xs[0]);
  });
  it('has a roughly standard normal distribution', () => {
    const g = gaussian(7);
    const n = 20000;
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < n; i++) {
      const x = g();
      s += x;
      s2 += x * x;
    }
    expect(Math.abs(s / n)).toBeLessThan(0.03);
    expect(Math.abs(s2 / n - 1)).toBeLessThan(0.05);
  });
  it('makes RGBA previews from latents', () => {
    const p = latentPreview(new Float32Array(4 * 8 * 8), 8);
    expect(p.data.length).toBe(8 * 8 * 4);
    expect(p.data[3]).toBe(255);
  });
});

describe('token offsets', () => {
  it('aligns WordPiece tokens', () => {
    const text = 'Ana works at Booking.com';
    const off = alignTokens(text, ['[CLS]', 'ana', 'works', 'at', 'book', '##ing', '.', 'com', '[SEP]'], true);
    expect(off[0]).toBeNull();
    expect(text.slice(...(off[1] as [number, number]))).toBe('Ana');
    expect(text.slice(...(off[5] as [number, number]))).toBe('ing');
    expect(off[8]).toBeNull();
  });
  it('decodes byte-level tokens into exact offsets', () => {
    const pieces = ['Hel', 'lo', ' wor', 'ld', '!'];
    const decode = (ids: number[]) => ids.map((i) => pieces[i]).join('');
    const text = 'Hello world!';
    const off = decodeOffsets(decode, [0, 1, 2, 3, 4], text);
    expect(off).toEqual([
      [0, 3],
      [3, 5],
      [6, 9],
      [9, 11],
      [11, 12],
    ]);
  });
  it('skips tokens that end mid-character', () => {
    const pieces = ['caf', '�', 'é'];
    let calls = 0;
    const decode = (ids: number[]) => {
      calls++;
      return ids.length === 1 ? 'caf' : ids.length === 2 ? 'caf�' : 'café';
    };
    const off = decodeOffsets(decode, [0, 1, 2], 'café');
    expect(off[1]).toBeNull();
    expect(calls).toBe(3);
    void pieces;
  });
});

describe('streamed text cleanup', () => {
  it('removes special tokens', () => {
    expect(stripSpecial('<|im_start|>Hi<|im_end|>')).toBe('Hi');
    expect(stripSpecial('<think>\n\n</think>\n\nAnswer')).toBe('Answer');
  });
  it('holds back partial special tokens between chunks', () => {
    const f = visibleTextFilter('lfm2', true);
    const out = ['Hi there', '<|im', '_end|>'].map((c) => f.push(c)).join('') + f.flush();
    expect(out).toBe('Hi there');
  });
});
