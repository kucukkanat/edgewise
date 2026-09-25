import { toEdgewiseError, UnsupportedDeviceError } from '../core/errors.ts';
import { type Message, messageText } from '../core/parts.ts';
import type { CommonOptions, RunInfo } from '../core/types.ts';
import { now } from '../core/util.ts';
import type { StepInput, StepOutput } from './lm.ts';

interface LanguageModelSession {
  promptStreaming(input: unknown, o?: { signal?: AbortSignal }): AsyncIterable<string> & ReadableStream<string>;
  destroy?(): void;
}
interface LanguageModelStatic {
  availability(o?: unknown): Promise<string>;
  create(o?: Record<string, unknown>): Promise<LanguageModelSession>;
}

/** Run a prompt on Chrome's built-in model through the Prompt API. */
export async function chromeStep(opts: CommonOptions, input: StepInput): Promise<{ out: StepOutput; info: RunInfo }> {
  const LM = (globalThis as { LanguageModel?: LanguageModelStatic }).LanguageModel;
  if (!LM) {
    throw new UnsupportedDeviceError('This browser has no built-in language model (the Prompt API).', {
      hint: "Pass a list such as ['chrome:gemini-nano', 'lfm2.5-350m'] to fall back to a downloaded model.",
    });
  }
  const avail = await LM.availability();
  if (avail === 'unavailable') throw new UnsupportedDeviceError('Chrome reports the built-in model as unavailable on this device.');
  const system = input.messages
    .filter((m) => m.role === 'system')
    .map(messageText)
    .join('\n');
  const convo = input.messages.filter((m: Message) => m.role === 'user' || m.role === 'assistant');
  const last = convo.pop();
  try {
    const session = await LM.create({
      initialPrompts: [...(system ? [{ role: 'system', content: system }] : []), ...convo.map((m) => ({ role: m.role, content: messageText(m) }))],
      signal: input.signal,
      monitor(mon: EventTarget) {
        mon.addEventListener('downloadprogress', (e: Event) => {
          const loaded = (e as unknown as { loaded: number }).loaded;
          opts.onProgress?.({ type: 'download', model: 'chrome:gemini-nano', file: 'built-in', loaded, total: 1 });
        });
      },
    });
    const t0 = now();
    let raw = '';
    const stream = session.promptStreaming(last ? messageText(last) : '', { signal: input.signal });
    for await (const chunk of stream as AsyncIterable<string>) {
      raw += chunk;
      input.onText?.(chunk);
    }
    session.destroy?.();
    const out: StepOutput = {
      raw,
      inputTokens: 0,
      outputTokens: Math.ceil(raw.length / 4),
      finishReason: input.signal?.aborted ? 'abort' : 'stop',
      seconds: (now() - t0) / 1000,
    };
    return { out, info: { model: 'chrome:gemini-nano', device: 'webgpu', dtype: 'built-in', backend: 'chrome-prompt-api' } };
  } catch (err) {
    throw toEdgewiseError(err, 'Chrome built-in model');
  }
}
