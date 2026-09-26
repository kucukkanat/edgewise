import { AbortError } from './errors.ts';

/** Every event a Run can emit on `run.events`. */
export type RunEvent =
  | { type: 'load'; event: import('./types.ts').LoadEvent }
  | { type: 'text-delta'; delta: string }
  | { type: 'object-delta'; partial: unknown }
  | { type: 'tool-call'; call: import('../verbs/generate.ts').ToolCall }
  | { type: 'tool-result'; result: import('../verbs/generate.ts').ToolResult }
  | { type: 'utterance-start' }
  | { type: 'utterance-end'; text: string }
  | { type: 'audio-chunk'; text: string; samples: number }
  | { type: 'step'; step: number; total: number }
  | { type: 'finish'; result: unknown };

export interface RunContext<C> {
  /** Push a chunk to iterators of the run. */
  emit(chunk: C): void;
  /** Push an event to `run.events`. */
  event(e: RunEvent): void;
  signal: AbortSignal;
}

/** A single-consumer async queue. */
class Channel<T> {
  private items: T[] = [];
  private waiters: ((r: IteratorResult<T>) => void)[] = [];
  private closed = false;
  private error: unknown = undefined;

  push(item: T): void {
    if (this.closed) return;
    const w = this.waiters.shift();
    if (w) w({ value: item, done: false });
    else this.items.push(item);
  }

  close(error?: unknown): void {
    if (this.closed) return;
    this.closed = true;
    this.error = error;
    for (const w of this.waiters.splice(0)) {
      if (error !== undefined) (w as unknown as (r: never) => void)(Promise.reject(error) as never);
      else w({ value: undefined as T, done: true });
    }
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.items.length) return { value: this.items.shift() as T, done: false };
    if (this.closed) {
      if (this.error !== undefined) throw this.error;
      return { value: undefined as T, done: true };
    }
    const r = await new Promise<IteratorResult<T>>((resolve) => this.waiters.push(resolve));
    return r;
  }
}

/**
 * A Run is both a promise of the final result and an async iterable of chunks.
 * It starts when you first await it, iterate it, or read `events`.
 *
 * ```ts
 * const run = generate({ model: 'text:tiny', input: 'Hi' });
 * for await (const delta of run) out.append(delta); // stream
 * const { usage } = await run;                       // then read the result
 * ```
 */
export class Run<Result, Chunk> implements PromiseLike<Result>, AsyncIterable<Chunk> {
  private started: Promise<Result> | null = null;
  private chunks = new Channel<Chunk>();
  private eventsChannel = new Channel<RunEvent>();
  private iterating = false;
  private eventsRead = false;
  private readonly controller = new AbortController();
  private detach: () => void = () => {};

  constructor(
    private readonly executor: (ctx: RunContext<Chunk>) => Promise<Result>,
    signal?: AbortSignal,
  ) {
    if (signal) {
      if (signal.aborted) this.controller.abort(signal.reason);
      else {
        const onAbort = () => this.controller.abort(signal.reason);
        signal.addEventListener('abort', onAbort, { once: true });
        // Drop the listener when the run settles, so a long-lived signal does not keep finished runs alive.
        this.detach = () => signal.removeEventListener('abort', onAbort);
      }
    }
  }

  private start(): Promise<Result> {
    if (this.started) return this.started;
    const ctx: RunContext<Chunk> = {
      emit: (c) => {
        if (this.iterating) this.chunks.push(c);
      },
      event: (e) => {
        if (this.eventsRead) this.eventsChannel.push(e);
      },
      signal: this.controller.signal,
    };
    const aborted = () => new AbortError(undefined, { cause: this.controller.signal.reason });
    this.started = (async () => {
      if (this.controller.signal.aborted) throw aborted();
      const r = await this.executor(ctx);
      // Every verb behaves the same way: a cancelled run rejects, even if the executor returned partial output.
      if (this.controller.signal.aborted) throw aborted();
      return r;
    })().then(
      (r) => {
        this.detach();
        ctx.event({ type: 'finish', result: r });
        this.chunks.close();
        this.eventsChannel.close();
        return r;
      },
      (err) => {
        this.detach();
        this.chunks.close(err);
        this.eventsChannel.close(err);
        throw err;
      },
    );
    return this.started;
  }

  then<A = Result, B = never>(
    onfulfilled?: ((value: Result) => A | PromiseLike<A>) | null,
    onrejected?: ((reason: unknown) => B | PromiseLike<B>) | null,
  ): Promise<A | B> {
    return this.start().then(onfulfilled, onrejected);
  }

  catch<B = never>(onrejected?: ((reason: unknown) => B | PromiseLike<B>) | null): Promise<Result | B> {
    return this.start().catch(onrejected);
  }

  finally(fn?: (() => void) | null): Promise<Result> {
    return this.start().finally(fn);
  }

  [Symbol.asyncIterator](): AsyncIterator<Chunk> {
    if (this.iterating) throw new Error('A Run can only be iterated once. Await it to read the result again.');
    this.iterating = true;
    const p = this.start();
    p.catch(() => {});
    return {
      next: () => this.chunks.next(),
      // Leaving a for-await loop early (break, return, throw) cancels the run.
      return: async () => {
        this.cancel(new AbortError('The consumer stopped reading the run.'));
        return { value: undefined as Chunk, done: true };
      },
    };
  }

  /** Every event, including tool calls, utterances and load progress. */
  get events(): AsyncIterable<RunEvent> {
    if (this.eventsRead) throw new Error('run.events can only be read once.');
    this.eventsRead = true;
    // Chunks are only buffered for iterators; mark iterating so the executor still streams.
    const p = this.start();
    p.catch(() => {});
    const ch = this.eventsChannel;
    return {
      [Symbol.asyncIterator]: () => ({ next: () => ch.next() }),
    };
  }

  /** Stop the run. Same as aborting the signal you passed. */
  cancel(reason?: unknown): void {
    this.controller.abort(reason ?? new AbortError('Run cancelled.'));
  }

  /** The signal the run listens to. */
  get signal(): AbortSignal {
    return this.controller.signal;
  }
}

export function createRun<R, C>(executor: (ctx: RunContext<C>) => Promise<R>, signal?: AbortSignal): Run<R, C> {
  return new Run(executor, signal);
}
