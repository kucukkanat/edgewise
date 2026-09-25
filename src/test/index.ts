/**
 * Test helpers: mock models that answer instantly, so app logic can be unit-tested without downloads.
 * @module
 */
import { defineModel } from '../core/registry.ts';
import type { Manifest, Verb } from '../core/types.ts';

export type MockGenerate = (input: {
  messages: import('../core/parts.ts').Message[];
}) =>
  | string
  | { text?: string; toolCalls?: { name: string; input: unknown }[] }
  | Promise<string | { text?: string; toolCalls?: { name: string; input: unknown }[] }>;

/** Per question name: a probability map for choice/score/boolean, or spans for spans(). */
export type MockEvaluate = (input: {
  state: string;
}) => Record<string, Record<string, number> | number | { type: string; start: number; end: number; score?: number }[]>;

export type MockEmbed = (input: { texts: string[] }) => Float32Array[] | number[][];

export type MockSpeak = (input: { text: string }) => Float32Array;

export type MockForecast = (input: { series: Float32Array; horizon: number }) => number[] | Float32Array;

export type MockOptions =
  | { verb: 'generate'; respond: MockGenerate; id?: string }
  | { verb: 'evaluate'; respond: MockEvaluate; id?: string }
  | { verb: 'embed'; respond: MockEmbed; id?: string; dimensions?: number }
  | { verb: 'speak'; respond?: MockSpeak; id?: string }
  | { verb: 'forecast'; respond: MockForecast; id?: string };

let seq = 0;

/** Create a mock model. Pass the returned manifest anywhere a model ID is accepted. */
export function mockModel(o: MockOptions): Manifest {
  const verb = o.verb as Verb;
  return defineModel({
    id: o.id ?? `mock:${verb}:${++seq}`,
    verb,
    accepts: verb === 'forecast' ? ['series'] : ['text', 'image', 'audio'],
    task: 'mock' as never,
    source: { url: 'mock://' },
    variants: [{ dtype: 'fp32', devices: ['webgpu', 'wasm', 'cpu'] }],
    status: 'stable',
    license: 'mit',
    features: ['tools', 'json', 'choice', 'score', 'boolean', 'spans', 'label'],
    config: { mock: o, toolFormat: 'hermes', dimensions: (o as { dimensions?: number }).dimensions, quantiles: [0.1, 0.5, 0.9] },
  });
}

export function isMock(m: Manifest): boolean {
  return (m.task as string) === 'mock';
}
