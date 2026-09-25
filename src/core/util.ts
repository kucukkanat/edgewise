/** Run tasks for the same key one at a time. ONNX sessions are not safe to run concurrently. */
const queues = new Map<string, Promise<unknown>>();

export function serialize<T>(key: string, fn: () => Promise<T>): Promise<T> {
  const prev = queues.get(key) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  const tail = next.catch(() => {});
  queues.set(key, tail);
  tail.then(() => {
    if (queues.get(key) === tail) queues.delete(key);
  });
  return next;
}

export function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export function softmax(xs: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > max) max = xs[i];
  const e: number[] = [];
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    const v = Math.exp(xs[i] - max);
    e.push(v);
    s += v;
  }
  return e.map((v) => v / s);
}

/** 1 minus normalized entropy: 1 when all mass is on one option, 0 when spread evenly. */
export function confidenceOf(probs: number[]): number {
  if (probs.length < 2) return 1;
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return Math.max(0, Math.min(1, 1 - h / Math.log(probs.length)));
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export function isAsyncIterable<T>(v: unknown): v is AsyncIterable<T> {
  return typeof v === 'object' && v !== null && typeof (v as AsyncIterable<T>)[Symbol.asyncIterator] === 'function';
}

export function toUint8(data: ArrayBuffer | Uint8Array): Uint8Array {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}
