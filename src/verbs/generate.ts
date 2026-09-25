import { chromeStep } from '../backends/chrome.ts';
import { type FlorencePreset, florence } from '../backends/florence.ts';
import { lmStep, type StepInput, type StepOutput, stripSpecial, visibleTextFilter } from '../backends/lm.ts';
import { toAudio } from '../backends/media.ts';
import { mockStep } from '../backends/mock.ts';
import { type Segment, transcribeSamples } from '../backends/stt.ts';
import { getConfig } from '../core/config.ts';
import { AbortError, ConfigError, SchemaValidationError, UnsupportedInputError } from '../core/errors.ts';
import { type Input, type Message, normalizeInput, type Part } from '../core/parts.ts';
import { registry, resolveManifest } from '../core/registry.ts';
import { createRun, type Run } from '../core/run.ts';
import { capabilities } from '../core/runtime.ts';
import type { CommonOptions, Manifest, ModelSpec, PartType, RunInfo, Usage } from '../core/types.ts';
import { extractJson, parsePartialJson, type SchemaLike, schemaInstruction, toJsonSchema, validateWith } from './schema.ts';
import { type AnyTool, parseToolCalls, type Tool, type ToolFormat, toolsForTemplate, validateCall } from './tools.ts';

export type { FlorencePreset } from '../backends/florence.ts';
export type { Segment } from '../backends/stt.ts';

export interface ToolCall {
  id: string;
  name: string;
  input: unknown;
}

export interface ToolResult {
  id: string;
  name: string;
  output?: unknown;
  error?: string;
}

export interface GenerateOptions<T = unknown> extends CommonOptions {
  /** Model ID, alias, manifest, or an ordered list to try. */
  model: ModelSpec;
  /** A single turn: a string, a part, media, or an array of these. */
  input?: Input;
  /** A conversation. Cannot be combined with `input`. */
  messages?: Message[];
  system?: string;
  /** Fill a Zod schema. The result's `object` is typed from it. */
  schema?: SchemaLike<T>;
  /** Tools the model may call. */
  tools?: Record<string, AnyTool>;
  /** Maximum model calls when tools are on. Default 3. */
  maxSteps?: number;
  /** Return false to decline a tool call before it runs. */
  approve?: (call: ToolCall) => boolean | Promise<boolean>;
  /** Fixed-prompt models such as Florence-2: caption, ocr, detect, or `{ find }`. */
  preset?: FlorencePreset;
  /** Speech models: return timestamps. */
  timestamps?: false | 'segment' | 'word';
  /** Speech models: spoken language, or 'auto'. */
  language?: string;
  /** Maximum new tokens per step. Default 512. */
  maxTokens?: number;
  /** 0 (default) decodes greedily. Higher values sample. */
  temperature?: number;
  topP?: number;
  /** Stop when the output contains one of these strings. */
  stop?: string[];
}

export interface GenerateResult<T = unknown> {
  text: string;
  object?: T;
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
  segments?: Segment[];
  language?: string;
  usage: Usage;
  finishReason: 'stop' | 'length' | 'tool-calls' | 'abort';
  /** The conversation including this reply, ready to pass back as `messages`. */
  messages: Message[];
  /** Which model, device and dtype ran. */
  info: RunInfo;
}

export type GenerateChunk<T> = string | (T extends object ? Partial<T> : T);

async function pickModel(spec: ModelSpec, types: PartType[], allowPreview?: boolean): Promise<Manifest> {
  const list = Array.isArray(spec) ? spec : [spec];
  if (!list.length) throw new ConfigError('model is an empty list.');
  let lastErr: unknown;
  for (const ref of list) {
    try {
      const m = resolveManifest(ref, { verb: 'generate', allowPreview, inputs: types });
      if (m.task === 'chrome-prompt') {
        const caps = await capabilities();
        if (caps.builtinAI === 'unavailable') throw new UnsupportedInputError('Chrome built-in AI is unavailable here.');
      }
      return m;
    } catch (err) {
      lastErr = err;
      if (list.length === 1) throw err;
    }
  }
  throw lastErr;
}

function lastUserParts(messages: Message[]): Part[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === 'user') return typeof m.content === 'string' ? [{ type: 'text', text: m.content }] : m.content;
  }
  return [];
}

const addUsage = (a: { inputTokens: number; outputTokens: number; seconds: number }, o: StepOutput) => {
  a.inputTokens += o.inputTokens;
  a.outputTokens += o.outputTokens;
  a.seconds += o.seconds;
};

/**
 * Generate text, or fill a schema, from any input the model accepts.
 * Await the Run for the result, or iterate it for text deltas.
 */
export function generate<T = unknown>(options: GenerateOptions<T>): Run<GenerateResult<T>, GenerateChunk<T>> {
  return createRun<GenerateResult<T>, GenerateChunk<T>>(async (ctx) => {
    const signal = ctx.signal;
    const norm = normalizeInput(options);
    const m = await pickModel(options.model, norm.types, options.allowPreview);
    const common: CommonOptions = {
      signal,
      device: options.device,
      dtype: options.dtype,
      allowPreview: options.allowPreview,
      onProgress: (e) => {
        options.onProgress?.(e);
        ctx.event({ type: 'load', event: e });
      },
    };
    const emitText = (d: string) => {
      if (!d) return;
      ctx.emit(d as GenerateChunk<T>);
      ctx.event({ type: 'text-delta', delta: d });
    };

    // ---------------------------------------------------------------- speech to text
    if (m.task === 'speech-to-text') {
      if (options.tools || options.schema) throw new ConfigError('Speech models do not support tools or schema.');
      const sttOpts = { timestamps: options.timestamps, language: options.language, signal, onText: emitText };
      if (norm.source) {
        const texts: string[] = [];
        let info: RunInfo | undefined;
        let tokens = 0;
        for await (const utterance of norm.source.utterances()) {
          if (signal.aborted) break;
          ctx.event({ type: 'utterance-start' });
          const r = await transcribeSamples(m, common, utterance, sttOpts);
          info = r.info;
          tokens += r.outputTokens;
          if (r.text) texts.push(r.text);
          ctx.event({ type: 'utterance-end', text: r.text });
          emitText(' ');
        }
        const text = texts.join(' ');
        return finish({ text, info: info ?? fallbackInfo(m), usage: { inputTokens: 0, outputTokens: tokens, tokensPerSecond: 0 }, messages: norm.messages });
      }
      const audio = lastUserParts(norm.messages).filter((p): p is Extract<Part, { type: 'audio' }> => p.type === 'audio');
      if (!audio.length) throw new UnsupportedInputError(`"${m.id}" needs audio input.`);
      const pieces: Float32Array[] = [];
      for (const a of audio) pieces.push(await toAudio(a.audio, a.sampleRate));
      const samples = concat(pieces);
      const r = await transcribeSamples(m, common, samples, sttOpts);
      return finish({
        text: r.text,
        segments: r.segments,
        language: r.language,
        info: r.info,
        usage: { inputTokens: 0, outputTokens: r.outputTokens, tokensPerSecond: 0 },
        messages: norm.messages,
      });
    }

    // ---------------------------------------------------------------- Florence presets
    if (m.task === 'florence2') {
      const parts = lastUserParts(norm.messages);
      const image = parts.find((p): p is Extract<Part, { type: 'image' }> => p.type === 'image');
      if (!image) throw new UnsupportedInputError(`"${m.id}" needs an image.`);
      const r = await florence(m, common, image.image, options.preset ?? 'caption', options.maxTokens ?? 256);
      emitText(r.result.text);
      return finish({
        text: r.result.text,
        object: r.result.object as T | undefined,
        info: r.info,
        usage: { inputTokens: r.inputTokens, outputTokens: r.outputTokens, tokensPerSecond: 0 },
        messages: norm.messages,
      });
    }

    if (options.preset) throw new ConfigError(`"${m.id}" has no presets. Presets work with florence-2-base.`);
    if (norm.source) throw new UnsupportedInputError(`"${m.id}" does not accept a live audio source.`);

    // ---------------------------------------------------------------- language models
    const format = m.config?.toolFormat as ToolFormat | undefined;
    const tools = options.tools && Object.keys(options.tools).length ? options.tools : undefined;
    if (tools && !format && m.task !== 'chrome-prompt' && m.task !== 'mock') {
      throw new ConfigError(`"${m.id}" does not support tool calling.`, {
        hint: `Use a model with tools, such as ${registry
          .list({ verb: 'generate' })
          .filter((x) => x.features?.includes('tools'))
          .map((x) => x.id)
          .slice(0, 3)
          .join(', ')}.`,
      });
    }
    const templateTools = tools ? await toolsForTemplate(tools) : undefined;
    const messages: Message[] = norm.messages.map((x) => ({ ...x }));
    let jsonSchema: Record<string, unknown> | undefined;
    if (options.schema) {
      jsonSchema = await toJsonSchema(options.schema as SchemaLike);
      const instr = schemaInstruction(jsonSchema);
      const sys = messages.find((x) => x.role === 'system');
      if (sys) sys.content = `${typeof sys.content === 'string' ? sys.content : ''}\n\n${instr}`;
      else messages.unshift({ role: 'system', content: instr });
    }
    const maxSteps = Math.max(1, options.maxSteps ?? 3);
    const usage = { inputTokens: 0, outputTokens: 0, seconds: 0 };
    const toolCalls: ToolCall[] = [];
    const toolResults: ToolResult[] = [];
    let info: RunInfo = fallbackInfo(m);
    let finishReason: GenerateResult['finishReason'] = 'stop';
    let finalRaw = '';
    let callSeq = 0;

    const step = async (msgs: Message[], streamVisible: boolean): Promise<StepOutput> => {
      const filter = visibleTextFilter(format, !!tools);
      let partialBuf = '';
      const onText = (chunk: string) => {
        if (options.schema) {
          partialBuf += chunk;
          const p = parsePartialJson(stripSpecial(partialBuf));
          if (p !== undefined && streamVisible) {
            ctx.emit(p as GenerateChunk<T>);
            ctx.event({ type: 'object-delta', partial: p });
          }
          return;
        }
        if (streamVisible) emitText(filter.push(chunk));
      };
      const input: StepInput = {
        messages: msgs,
        tools: templateTools,
        maxTokens: options.maxTokens ?? 512,
        temperature: options.temperature,
        topP: options.topP,
        stop: options.stop,
        signal,
        onText,
      };
      const r = m.task === 'mock' ? await mockStep(m, input) : m.task === 'chrome-prompt' ? await chromeStep(common, input) : await lmStep(m, common, input);
      if (!options.schema && streamVisible) emitText(filter.flush());
      info = r.info;
      addUsage(usage, r.out);
      return r.out;
    };

    for (let i = 0; i < maxSteps; i++) {
      const out = await step(messages, true);
      finalRaw = out.raw;
      finishReason = out.finishReason === 'abort' ? 'abort' : out.finishReason;
      if (signal.aborted) {
        finishReason = 'abort';
        break;
      }
      if (!tools || !format) break;
      const parsed = parseToolCalls(out.raw, format);
      if (!parsed.calls.length) break;
      messages.push({ role: 'assistant', content: out.raw.replace(/<\|im_end\|>\s*$/, '') });
      const pending: ToolCall[] = [];
      for (const rc of parsed.calls) {
        const call: ToolCall = { id: `call_${++callSeq}`, name: rc.name, input: rc.input };
        const v = validateCall(tools as Record<string, Tool>, rc);
        if (v.ok) call.input = v.input;
        toolCalls.push(call);
        ctx.event({ type: 'tool-call', call });
        let result: ToolResult;
        if (!v.ok) result = { id: call.id, name: call.name, error: v.error };
        else if (options.approve && !(await options.approve(call))) {
          result = { id: call.id, name: call.name, error: 'The user declined this action.' };
        } else {
          const t = tools[call.name];
          if (!t.execute) {
            pending.push(call);
            continue;
          }
          try {
            const output = await t.execute(call.input as never, { signal, call: { id: call.id, name: call.name } });
            result = { id: call.id, name: call.name, output };
          } catch (err) {
            result = { id: call.id, name: call.name, error: err instanceof Error ? err.message : String(err) };
          }
        }
        toolResults.push(result);
        ctx.event({ type: 'tool-result', result });
        messages.push({
          role: 'tool',
          name: call.name,
          toolCallId: call.id,
          content: JSON.stringify(result.error ? { error: result.error } : (result.output ?? null)),
        });
      }
      if (pending.length) {
        finishReason = 'tool-calls';
        break;
      }
      if (i === maxSteps - 1) finishReason = 'tool-calls';
    }

    const parsedFinal = tools && format ? parseToolCalls(finalRaw, format) : { calls: [], text: finalRaw };
    let text = stripSpecial(parsedFinal.text).trim();
    let object: T | undefined;
    if (options.schema && finishReason !== 'abort') {
      const tryParse = (s: string) => {
        try {
          const v = extractJson(s);
          const r = validateWith(options.schema as SchemaLike<T>, v);
          return r.success ? { ok: true as const, value: r.data } : { ok: false as const, error: r.error, issues: (r as { issues?: unknown }).issues };
        } catch (err) {
          return { ok: false as const, error: `Not valid JSON: ${err instanceof Error ? err.message : String(err)}` };
        }
      };
      let r = tryParse(text);
      if (!r.ok) {
        const retry: Message[] = [
          ...messages,
          { role: 'assistant', content: text },
          { role: 'user', content: `That reply did not match the schema (${r.error}). Reply again with only the corrected JSON.` },
        ];
        const out2 = await step(retry, false);
        text = stripSpecial(out2.raw).trim();
        r = tryParse(text);
      }
      if (!r.ok) {
        throw new SchemaValidationError(`The model's reply did not match the schema: ${r.error}`, {
          raw: text,
          issues: (r as { issues?: unknown }).issues,
          hint: 'Try a larger model, simplify the schema, or describe the fields in the prompt.',
        });
      }
      object = r.value;
    }
    messages.push({ role: 'assistant', content: text });
    return finish({
      text,
      object,
      toolCalls,
      toolResults,
      info,
      finishReason,
      usage: {
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        tokensPerSecond: usage.seconds > 0 ? Math.round((usage.outputTokens / usage.seconds) * 10) / 10 : 0,
      },
      messages,
    });

    function finish(r: Partial<GenerateResult<T>> & Pick<GenerateResult<T>, 'text' | 'info' | 'usage' | 'messages'>): GenerateResult<T> {
      if (signal.aborted && !r.finishReason) throw new AbortError(undefined, { cause: signal.reason });
      return {
        toolCalls: [],
        toolResults: [],
        finishReason: 'stop',
        ...r,
      } as GenerateResult<T>;
    }
  }, options.signal);
}

function fallbackInfo(m: Manifest): RunInfo {
  return { model: m.id, device: 'cpu', dtype: 'unknown', backend: 'none' };
}

function concat(parts: Float32Array[]): Float32Array {
  if (parts.length === 1) return parts[0];
  const n = parts.reduce((a, b) => a + b.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

export const _test = { getConfig };
