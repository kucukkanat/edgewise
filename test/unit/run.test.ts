import { AbortError } from '../../src/core/errors.ts';
import { Run } from '../../src/core/run.ts';

function counter(n: number, delay = 1) {
  return new Run<number, number>(async (ctx) => {
    let sum = 0;
    for (let i = 1; i <= n; i++) {
      if (ctx.signal.aborted) throw new AbortError();
      await new Promise((r) => setTimeout(r, delay));
      ctx.emit(i);
      ctx.event({ type: 'text-delta', delta: String(i) });
      sum += i;
    }
    return sum;
  });
}

describe('Run', () => {
  it('can be awaited', async () => {
    expect(await counter(4)).toBe(10);
  });

  it('can be iterated, then awaited without running again', async () => {
    const r = counter(3);
    const seen: number[] = [];
    for await (const x of r) seen.push(x);
    expect(seen).toEqual([1, 2, 3]);
    expect(await r).toBe(6);
  });

  it('exposes events', async () => {
    const r = counter(2);
    const types: string[] = [];
    for await (const e of r.events) types.push(e.type);
    expect(types).toEqual(['text-delta', 'text-delta', 'finish']);
  });

  it('refuses a second iterator', () => {
    const r = counter(1);
    r[Symbol.asyncIterator]();
    expect(() => r[Symbol.asyncIterator]()).toThrow(/once/);
  });

  it('cancels', async () => {
    const r = counter(100, 5);
    setTimeout(() => r.cancel(), 12);
    await expect(Promise.resolve(r)).rejects.toBeInstanceOf(AbortError);
  });

  it('honours an already aborted signal', async () => {
    const ac = new AbortController();
    ac.abort();
    const r = new Run<number, number>(async () => 1, ac.signal);
    await expect(Promise.resolve(r)).rejects.toBeInstanceOf(AbortError);
  });

  it('propagates errors to iterators', async () => {
    const r = new Run<number, number>(async (ctx) => {
      ctx.emit(1);
      await new Promise((res) => setTimeout(res, 1));
      throw new Error('bad');
    });
    const seen: number[] = [];
    await expect(
      (async () => {
        for await (const x of r) seen.push(x);
      })(),
    ).rejects.toThrow('bad');
    expect(seen).toEqual([1]);
  });

  it('works with Promise.all', async () => {
    const [a, b] = await Promise.all([counter(2), counter(3)]);
    expect([a, b]).toEqual([3, 6]);
  });
});
