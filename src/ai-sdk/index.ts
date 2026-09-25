/**
 * Use Edgewise models with the Vercel AI SDK (spec v4), and send Edgewise questions to cloud judges.
 * @module
 */
import type {
  EmbeddingModelV4,
  Experimental_EvaluationModelV4 as EvaluationModelV4,
  Experimental_EvaluationModelV4Question as EvaluationQuestion,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4StreamPart,
  LanguageModelV4Usage,
} from '@ai-sdk/provider';
import { ConfigError } from '../core/errors.ts';
import type { Message, Part } from '../core/parts.ts';
import { registry } from '../core/registry.ts';
import type { ModelRef } from '../core/types.ts';
import { embed } from '../verbs/embed.ts';
import { type BooleanQuestion, type ChoiceQuestion, evaluate, type Question, type ScoreQuestion } from '../verbs/evaluate.ts';
import { type GenerateResult, generate } from '../verbs/generate.ts';
import type { AnyTool } from '../verbs/tools.ts';

function toBlob(data: unknown, mediaType: string): Blob | URL | string {
  if (data instanceof URL) return data;
  if (data instanceof Uint8Array) return new Blob([data as Uint8Array<ArrayBuffer>], { type: mediaType });
  if (typeof data === 'string') {
    if (/^https?:\/\//.test(data)) return new URL(data);
    const bin = atob(data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: mediaType });
  }
  if (data && typeof data === 'object' && 'data' in (data as object)) return toBlob((data as { data: unknown }).data, mediaType);
  throw new ConfigError('Unsupported file data in AI SDK prompt.');
}

/** Convert an AI SDK prompt into Edgewise messages. */
export function fromPrompt(prompt: LanguageModelV4CallOptions['prompt']): Message[] {
  const out: Message[] = [];
  for (const m of prompt) {
    if (m.role === 'system') out.push({ role: 'system', content: m.content });
    else if (m.role === 'user') {
      const parts: Part[] = [];
      for (const p of m.content) {
        if (p.type === 'text') parts.push({ type: 'text', text: p.text });
        else if (p.type === 'file') {
          const d = toBlob(p.data, p.mediaType) as Blob;
          if (p.mediaType.startsWith('image/')) parts.push({ type: 'image', image: d });
          else if (p.mediaType.startsWith('audio/')) parts.push({ type: 'audio', audio: d });
        }
      }
      out.push({ role: 'user', content: parts });
    } else if (m.role === 'assistant') {
      const text = m.content
        .map((p) =>
          p.type === 'text' ? p.text : p.type === 'tool-call' ? `<tool_call>${JSON.stringify({ name: p.toolName, arguments: p.input })}</tool_call>` : '',
        )
        .join('');
      out.push({ role: 'assistant', content: text });
    } else if (m.role === 'tool') {
      for (const p of m.content) {
        if (p.type !== 'tool-result') continue;
        const o = p.output as { type: string; value?: unknown; reason?: string };
        const value = o.type === 'execution-denied' ? { error: o.reason ?? 'denied' } : o.value;
        out.push({ role: 'tool', name: p.toolName, toolCallId: p.toolCallId, content: typeof value === 'string' ? value : JSON.stringify(value) });
      }
    }
  }
  return out;
}

function toolsOf(opts: LanguageModelV4CallOptions): Record<string, AnyTool> | undefined {
  const list = (opts.tools ?? []).filter((t) => t.type === 'function') as { name: string; description?: string; inputSchema: Record<string, unknown> }[];
  if (!list.length || opts.toolChoice?.type === 'none') return undefined;
  return Object.fromEntries(list.map((t) => [t.name, { description: t.description ?? t.name, input: { jsonSchema: t.inputSchema } }]));
}

function usageOf(r: GenerateResult): LanguageModelV4Usage {
  return {
    inputTokens: { total: r.usage.inputTokens, noCache: r.usage.inputTokens, cacheRead: undefined, cacheWrite: undefined },
    outputTokens: { total: r.usage.outputTokens, text: r.usage.outputTokens, reasoning: undefined },
  };
}

function finishOf(r: GenerateResult): LanguageModelV4FinishReason {
  const map = { stop: 'stop', length: 'length', 'tool-calls': 'tool-calls', abort: 'other' } as const;
  return { unified: map[r.finishReason], raw: r.finishReason };
}

function callOpts(opts: LanguageModelV4CallOptions) {
  const schema =
    opts.responseFormat?.type === 'json' && opts.responseFormat.schema ? { jsonSchema: opts.responseFormat.schema as Record<string, unknown> } : undefined;
  return {
    messages: fromPrompt(opts.prompt),
    tools: toolsOf(opts),
    maxSteps: 1,
    maxTokens: opts.maxOutputTokens,
    temperature: opts.temperature,
    topP: opts.topP,
    stop: opts.stopSequences,
    schema,
    signal: opts.abortSignal,
  };
}

function content(r: GenerateResult): LanguageModelV4Content[] {
  const out: LanguageModelV4Content[] = [];
  if (r.text) out.push({ type: 'text', text: r.object !== undefined ? JSON.stringify(r.object) : r.text });
  for (const c of r.toolCalls) out.push({ type: 'tool-call', toolCallId: c.id, toolName: c.name, input: JSON.stringify(c.input) });
  return out;
}

/** An Edgewise `generate` model as an AI SDK language model. */
export function edgewiseLanguageModel(model: ModelRef): LanguageModelV4 {
  const id = typeof model === 'string' ? model : model.id;
  return {
    specificationVersion: 'v4',
    provider: 'edgewise',
    modelId: id,
    supportedUrls: { 'image/*': [/^https?:\/\//], 'audio/*': [/^https?:\/\//] },
    async doGenerate(opts) {
      const r = await generate({ model, ...callOpts(opts) });
      return { content: content(r), finishReason: finishOf(r), usage: usageOf(r), warnings: [], response: { modelId: r.info.model } };
    },
    async doStream(opts) {
      const run = generate({ model, ...callOpts(opts) });
      const stream = new ReadableStream<LanguageModelV4StreamPart>({
        async start(controller) {
          controller.enqueue({ type: 'stream-start', warnings: [] });
          controller.enqueue({ type: 'text-start', id: 't0' });
          try {
            for await (const d of run) if (typeof d === 'string') controller.enqueue({ type: 'text-delta', id: 't0', delta: d });
            const r = await run;
            controller.enqueue({ type: 'text-end', id: 't0' });
            for (const c of r.toolCalls) controller.enqueue({ type: 'tool-call', toolCallId: c.id, toolName: c.name, input: JSON.stringify(c.input) });
            controller.enqueue({ type: 'finish', usage: usageOf(r), finishReason: finishOf(r) });
          } catch (err) {
            controller.enqueue({ type: 'error', error: err });
          }
          controller.close();
        },
      });
      return { stream };
    },
  };
}

/** An Edgewise `embed` model as an AI SDK embedding model. */
export function edgewiseEmbeddingModel(model: ModelRef, o: { dimensions?: number } = {}): EmbeddingModelV4 {
  const id = typeof model === 'string' ? model : model.id;
  return {
    specificationVersion: 'v4',
    provider: 'edgewise',
    modelId: id,
    maxEmbeddingsPerCall: 256,
    supportsParallelCalls: false,
    async doEmbed({ values, abortSignal }) {
      const r = await embed({ model, values, dimensions: o.dimensions, signal: abortSignal });
      return { embeddings: r.embeddings.map((v) => Array.from(v)), warnings: [] };
    },
  } as EmbeddingModelV4;
}

function fromAiQuestion(q: EvaluationQuestion): Question {
  const text = (v: unknown) => (v == null ? '' : typeof v === 'string' ? v : JSON.stringify(v));
  if (q.type === 'choice') return { kind: 'choice', options: Object.fromEntries(Object.entries(q.criteria).map(([k, v]) => [k, text(v) || k])) };
  if (q.type === 'score') return { kind: 'score', levels: q.criteria.map((c, i) => text(c) || String(i)) };
  return { kind: 'boolean', true: text(q.criteria?.true) || text(q.instructions), false: text(q.criteria?.false) || undefined };
}

/** An Edgewise `evaluate` model as an AI SDK evaluation model (for `experimental_evaluate`). */
export function edgewiseEvaluationModel(model: ModelRef): EvaluationModelV4 {
  const id = typeof model === 'string' ? model : model.id;
  return {
    specificationVersion: 'v4',
    provider: 'edgewise',
    modelId: id,
    supportedQuestionTypes: ['choice', 'score', 'boolean'],
    async doEvaluate({ state, questions, abortSignal }) {
      const qs = Object.fromEntries(Object.entries(questions).map(([k, q]) => [k, fromAiQuestion(q)]));
      const r = await evaluate({ model, state, questions: qs, signal: abortSignal });
      const answers: Record<string, unknown> = {};
      for (const [k, q] of Object.entries(qs)) {
        const a = r.answers[k] as unknown as Record<string, unknown>;
        if (q.kind === 'choice') answers[k] = { type: 'choice', choice: a.choice, probabilities: a.probabilities };
        else if (q.kind === 'score') {
          const probs = a.probabilities as Record<string, number>;
          answers[k] = { type: 'score', score: a.score, probabilities: Object.fromEntries(Object.values(probs).map((p, i) => [String(i), p])) };
        } else answers[k] = { type: 'boolean', probability: a.probability };
      }
      return { answers, warnings: [] } as Awaited<ReturnType<EvaluationModelV4['doEvaluate']>>;
    },
  };
}

/** Pick the right AI SDK model type for an Edgewise model by its verb. */
export function edgewise(model: ModelRef): LanguageModelV4 | EmbeddingModelV4 | EvaluationModelV4 {
  const m = typeof model === 'string' ? registry.get(model) : model;
  if (m.verb === 'generate') return edgewiseLanguageModel(model);
  if (m.verb === 'embed') return edgewiseEmbeddingModel(model);
  if (m.verb === 'evaluate') return edgewiseEvaluationModel(model);
  throw new ConfigError(`"${m.id}" is a ${m.verb} model; the AI SDK adapter covers generate, embed and evaluate.`);
}

/** Convert Edgewise questions to AI SDK evaluation questions, for cloud judges such as Jev. */
export function toAiSdkQuestions(questions: Record<string, Question>): Record<string, EvaluationQuestion> {
  const out: Record<string, EvaluationQuestion> = {};
  for (const [name, q] of Object.entries(questions)) {
    const instructions = q.instructions ?? name.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[_-]/g, ' ');
    if (q.kind === 'choice') out[name] = { type: 'choice', instructions, criteria: (q as ChoiceQuestion).options };
    else if (q.kind === 'score') out[name] = { type: 'score', instructions, criteria: [...(q as ScoreQuestion).levels] };
    else if (q.kind === 'boolean') {
      const b = q as BooleanQuestion;
      out[name] = { type: 'boolean', instructions, criteria: { true: b.true ?? null, false: b.false ?? null } };
    } else throw new ConfigError(`${q.kind}() questions have no AI SDK equivalent ("${name}").`);
  }
  return out;
}
