import type { Message } from '../core/parts.ts';
import type { Manifest, RunInfo } from '../core/types.ts';
import type { StepInput, StepOutput } from './lm.ts';

const info = (m: Manifest): RunInfo => ({ model: m.id, device: 'cpu', dtype: 'mock', backend: 'mock' });

// biome-ignore lint/suspicious/noExplicitAny: mock callbacks are typed at the public API
type Any = any;

export async function mockStep(m: Manifest, input: StepInput): Promise<{ out: StepOutput; info: RunInfo }> {
  const o = m.config?.mock as Any;
  const r = await o.respond({ messages: [...(input.messages as Message[])] });
  const text: string = typeof r === 'string' ? r : (r.text ?? '');
  const calls = typeof r === 'string' ? [] : (r.toolCalls ?? []);
  let raw = '';
  for (const word of text.split(/(?<=\s)/)) {
    if (input.signal?.aborted) break;
    raw += word;
    input.onText?.(word);
  }
  for (const c of calls) raw += `<tool_call>${JSON.stringify({ name: c.name, arguments: c.input })}</tool_call>`;
  return {
    out: { raw, inputTokens: 0, outputTokens: raw.split(/\s+/).length, finishReason: input.signal?.aborted ? 'abort' : 'stop', seconds: 0.001 },
    info: info(m),
  };
}

export function mockEvaluate(m: Manifest, name: string, state: string): { value: Any; info: RunInfo } {
  const o = m.config?.mock as Any;
  return { value: o.respond({ state })[name], info: info(m) };
}

export function mockEmbed(m: Manifest, texts: string[]): { vectors: Float32Array[]; info: RunInfo } {
  const o = m.config?.mock as Any;
  const v = o.respond({ texts }) as (Float32Array | number[])[];
  return { vectors: v.map((x) => (x instanceof Float32Array ? x : Float32Array.from(x))), info: info(m) };
}

export function mockSpeak(m: Manifest, text: string): { samples: Float32Array; info: RunInfo } {
  const o = m.config?.mock as Any;
  return { samples: o.respond ? o.respond({ text }) : new Float32Array(Math.ceil(text.length * 240)), info: info(m) };
}

export function mockForecast(m: Manifest, series: Float32Array, horizon: number): { quantiles: Float32Array[]; info: RunInfo } {
  const o = m.config?.mock as Any;
  const med = Float32Array.from(o.respond({ series, horizon }) as number[]);
  return { quantiles: [med, med, med], info: info(m) };
}
