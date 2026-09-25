import { UnsupportedInputError } from '../core/errors.ts';
import { type Message, messageText, type Part } from '../core/parts.ts';
import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { now, serialize } from '../core/util.ts';
import { MarkupFilter, type ToolFormat, toolMarkers } from '../verbs/tools.ts';
import { toRawImage } from './media.ts';
import { getTransformers, loadTjs } from './transformers.ts';

type TJS = typeof import('@huggingface/transformers');

export interface StepInput {
  messages: Message[];
  tools?: unknown[];
  maxTokens: number;
  temperature?: number;
  topP?: number;
  stop?: string[];
  signal?: AbortSignal;
  onText?: (delta: string) => void;
}

export interface StepOutput {
  raw: string;
  inputTokens: number;
  outputTokens: number;
  finishReason: 'stop' | 'length' | 'abort';
  seconds: number;
}

const SPECIAL = /<\|[^|>]{1,40}\|>|<\/?s>|<eos>|<bos>|<pad>|<end_of_turn>|<start_of_turn>(model|user)?\n?|<think>\s*<\/think>\s*/g;

export function stripSpecial(s: string): string {
  return s.replace(SPECIAL, '');
}

interface LmBundle {
  kind: 'lm';
  tokenizer: InstanceType<TJS['PreTrainedTokenizer']>;
  model: InstanceType<TJS['PreTrainedModel']>;
}
interface VlmBundle {
  kind: 'vlm';
  processor: Awaited<ReturnType<TJS['AutoProcessor']['from_pretrained']>>;
  model: InstanceType<TJS['PreTrainedModel']>;
}

export async function loadLm(m: Manifest, opts: CommonOptions) {
  return loadTjs<LmBundle>(
    m,
    opts,
    async (t, base) => {
      const [tokenizer, model] = await Promise.all([
        t.AutoTokenizer.from_pretrained(m.source && 'repo' in m.source ? m.source.repo : '', {
          revision: base.revision,
          progress_callback: base.progress_callback as never,
        }),
        t.AutoModelForCausalLM.from_pretrained('repo' in m.source ? m.source.repo : '', base as never),
      ]);
      return { kind: 'lm', tokenizer, model } as LmBundle;
    },
    async (b) => {
      await (b.model as unknown as { dispose?: () => Promise<void> }).dispose?.();
    },
  );
}

export async function loadVlm(m: Manifest, opts: CommonOptions) {
  return loadTjs<VlmBundle>(
    m,
    opts,
    async (t, base) => {
      const repo = 'repo' in m.source ? m.source.repo : '';
      const [processor, model] = await Promise.all([
        t.AutoProcessor.from_pretrained(repo, { revision: base.revision, progress_callback: base.progress_callback as never }),
        t.AutoModelForImageTextToText.from_pretrained(repo, base as never),
      ]);
      return { kind: 'vlm', processor, model } as VlmBundle;
    },
    async (b) => {
      await (b.model as unknown as { dispose?: () => Promise<void> }).dispose?.();
    },
  );
}

/** Messages in the shape chat templates expect. Images become `{ type: 'image' }` placeholders. */
async function templateMessages(
  t: TJS,
  messages: Message[],
  withImages: boolean,
): Promise<{ msgs: Record<string, unknown>[]; images: InstanceType<TJS['RawImage']>[] }> {
  const images: InstanceType<TJS['RawImage']>[] = [];
  const msgs: Record<string, unknown>[] = [];
  for (const msg of messages) {
    if (!withImages || typeof msg.content === 'string') {
      const text = messageText(msg);
      const out: Record<string, unknown> = { role: msg.role, content: text };
      if (msg.role === 'tool' && msg.name) out.name = msg.name;
      msgs.push(out);
      continue;
    }
    const content: Record<string, unknown>[] = [];
    for (const p of msg.content as Part[]) {
      if (p.type === 'text') content.push({ type: 'text', text: p.text });
      else if (p.type === 'image') {
        images.push(await toRawImage(t, p.image));
        content.push({ type: 'image' });
      } else if (p.type === 'video') {
        for (const f of await videoFrames(p)) {
          images.push(await toRawImage(t, f));
          content.push({ type: 'image' });
        }
      } else if (p.type === 'audio') {
        throw new UnsupportedInputError('This model does not accept audio.');
      }
    }
    msgs.push({ role: msg.role, content });
  }
  return { msgs, images };
}

async function videoFrames(p: Extract<Part, { type: 'video' }>): Promise<import('../core/parts.ts').ImageLike[]> {
  const n = Math.max(1, Math.min(32, p.frames ?? 8));
  if (Array.isArray(p.video)) {
    const frames = p.video;
    if (frames.length <= n) return frames;
    return Array.from({ length: n }, (_, i) => frames[Math.floor((i * frames.length) / n)]);
  }
  if (typeof document === 'undefined') {
    throw new UnsupportedInputError('Video decoding needs a browser.', {
      hint: "On Bun and Node, pass { type: 'video', video: [frame1, frame2, …] } with frames you decoded yourself.",
    });
  }
  const el = document.createElement('video');
  el.muted = true;
  el.playsInline = true;
  const owned = p.video instanceof Blob;
  if (owned) el.src = URL.createObjectURL(p.video as Blob);
  else {
    const v = p.video as HTMLVideoElement;
    el.src = v.currentSrc || v.src;
  }
  await new Promise<void>((res, rej) => {
    el.onloadedmetadata = () => res();
    el.onerror = () => rej(new UnsupportedInputError('The browser could not decode this video.'));
  });
  const out: ImageBitmap[] = [];
  for (let i = 0; i < n; i++) {
    el.currentTime = ((i + 0.5) / n) * el.duration;
    await new Promise<void>((r) => {
      el.onseeked = () => r();
    });
    out.push(await createImageBitmap(el));
  }
  if (owned) URL.revokeObjectURL(el.src);
  return out;
}

/** Run one generation step (no tool loop) on a causal LM or a vision-language model. */
export async function lmStep(m: Manifest, opts: CommonOptions, input: StepInput): Promise<{ out: StepOutput; info: RunInfo }> {
  const t = await getTransformers();
  const vlm = m.task === 'image-text-to-text';
  const loaded = vlm ? await loadVlm(m, opts) : await loadLm(m, opts);
  const bundle = loaded.value;
  const tokenizer = bundle.kind === 'lm' ? bundle.tokenizer : (bundle.processor as unknown as { tokenizer: LmBundle['tokenizer'] }).tokenizer;
  const templateOpts = (m.config?.chatTemplateOptions as Record<string, unknown> | undefined) ?? {};
  const { msgs, images } = await templateMessages(t, input.messages, vlm);

  const out = await serialize(`gen:${loaded.info.model}:${loaded.info.device}`, async () => {
    let inputs: Record<string, unknown>;
    if (bundle.kind === 'vlm') {
      const proc = bundle.processor as unknown as {
        apply_chat_template: (m: unknown, o: unknown) => string;
        (images: unknown, text: string, o?: unknown): Promise<Record<string, unknown>>;
      };
      const text = proc.apply_chat_template(msgs, { add_generation_prompt: true, tools: input.tools, ...templateOpts });
      inputs = images.length
        ? await proc(images.length === 1 ? images[0] : images, text)
        : (tokenizer(text, { add_special_tokens: false }) as unknown as Record<string, unknown>);
    } else {
      const prompt = tokenizer.apply_chat_template(msgs as never, {
        tools: input.tools as never,
        add_generation_prompt: true,
        tokenize: false,
        ...templateOpts,
      }) as unknown as string;
      inputs = tokenizer(prompt, { add_special_tokens: false }) as unknown as Record<string, unknown>;
    }
    const inputIds = inputs.input_ids as { dims: number[] };
    const inputTokens = inputIds.dims.at(-1) ?? 0;
    const stopper = new t.InterruptableStoppingCriteria();
    let aborted = false;
    const onAbort = () => {
      aborted = true;
      stopper.interrupt();
    };
    input.signal?.addEventListener('abort', onAbort, { once: true });
    if (input.signal?.aborted) onAbort();
    let raw = '';
    let hitStop = false;
    let outputTokens = 0;
    const streamer = new t.TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: false,
      callback_function: (chunk: string) => {
        raw += chunk;
        if (input.stop?.length) {
          for (const s of input.stop) {
            const at = raw.indexOf(s);
            if (at >= 0) {
              raw = raw.slice(0, at);
              hitStop = true;
              stopper.interrupt();
              return;
            }
          }
        }
        input.onText?.(chunk);
      },
      token_callback_function: () => {
        outputTokens++;
      },
    } as never);
    const t0 = now();
    const sample = (input.temperature ?? 0) > 0;
    const seq = (await bundle.model.generate({
      ...inputs,
      max_new_tokens: input.maxTokens,
      do_sample: sample,
      ...(sample ? { temperature: input.temperature, top_p: input.topP ?? 0.95 } : {}),
      repetition_penalty: 1.05,
      streamer,
      stopping_criteria: [stopper],
    } as never)) as { dims: number[]; slice: (...a: unknown[]) => unknown };
    input.signal?.removeEventListener('abort', onAbort);
    const total = seq.dims.at(-1) ?? inputTokens;
    const generated = Math.max(outputTokens, total - inputTokens);
    if (!hitStop) {
      const ids = seq.slice(null, [inputTokens, null]);
      const decoded = tokenizer.batch_decode(ids as never, { skip_special_tokens: false })[0] ?? raw;
      if (decoded.length >= raw.length) raw = decoded;
    }
    const finishReason: StepOutput['finishReason'] = aborted ? 'abort' : hitStop ? 'stop' : generated >= input.maxTokens ? 'length' : 'stop';
    return { raw, inputTokens, outputTokens: generated, finishReason, seconds: (now() - t0) / 1000 };
  });
  return { out, info: loaded.info };
}

/** Streamed deltas cleaned of special tokens and, when tools are on, of tool-call markup. */
export function visibleTextFilter(format: ToolFormat | undefined, tools: boolean): { push: (chunk: string) => string; flush: () => string } {
  const markup = tools && format ? new MarkupFilter(toolMarkers(format).start, toolMarkers(format).end) : null;
  let pending = '';
  return {
    push(chunk: string) {
      const s = markup ? markup.push(chunk) : chunk;
      pending += s;
      // Hold back a possible partial special token at the end.
      const cut = pending.lastIndexOf('<');
      let emit = pending;
      if (cut >= 0 && !pending.slice(cut).includes('>') && pending.length - cut < 40) {
        emit = pending.slice(0, cut);
        pending = pending.slice(cut);
      } else pending = '';
      return stripSpecial(emit);
    },
    flush() {
      const rest = pending + (markup ? markup.flush() : '');
      pending = '';
      return stripSpecial(rest);
    },
  };
}
