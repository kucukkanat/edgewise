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

describe('Run cancellation semantics', () => {
  it('rejects a run cancelled part-way, even when the executor returns partial output', async () => {
    const r = new Run<string, number>(async (ctx) => {
      for (let i = 0; i < 10; i++) {
        if (ctx.signal.aborted) break;
        ctx.emit(i);
        await new Promise((res) => setTimeout(res, 5));
      }
      return 'partial';
    });
    setTimeout(() => r.cancel(), 12);
    await expect(Promise.resolve(r)).rejects.toBeInstanceOf(AbortError);
  });

  it('cancels when the consumer breaks out of for-await', async () => {
    const r = counter(100, 2);
    for await (const n of r) if (n >= 2) break;
    expect(r.signal.aborted).toBe(true);
    await expect(Promise.resolve(r)).rejects.toBeInstanceOf(AbortError);
  });

  it('removes its listener from a shared signal once settled', async () => {
    const ac = new AbortController();
    let added = 0;
    let removed = 0;
    const add = ac.signal.addEventListener.bind(ac.signal);
    const rem = ac.signal.removeEventListener.bind(ac.signal);
    ac.signal.addEventListener = ((...a: Parameters<typeof add>) => (added++, add(...a))) as typeof add;
    ac.signal.removeEventListener = ((...a: Parameters<typeof rem>) => (removed++, rem(...a))) as typeof rem;
    for (let i = 0; i < 5; i++) await new Run<number, never>(async () => i, ac.signal);
    expect(added).toBe(5);
    expect(removed).toBe(5);
  });
});
