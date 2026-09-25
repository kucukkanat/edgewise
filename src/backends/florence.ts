import { ConfigError } from '../core/errors.ts';
import type { ImageLike } from '../core/parts.ts';
import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { serialize } from '../core/util.ts';
import { toRawImage } from './media.ts';
import { getTransformers, loadTjs, repoOf } from './transformers.ts';

export type FlorencePreset = 'caption' | 'caption-detailed' | 'caption-more-detailed' | 'ocr' | 'detect' | { find: string };

export interface Box {
  label: string;
  box: [number, number, number, number];
}

export interface FlorenceResult {
  text: string;
  object?: { boxes?: Box[]; regions?: { text: string; box: [number, number, number, number] }[] };
}

const TOKENS: Record<Exclude<FlorencePreset, { find: string }>, string> = {
  caption: '<CAPTION>',
  'caption-detailed': '<DETAILED_CAPTION>',
  'caption-more-detailed': '<MORE_DETAILED_CAPTION>',
  ocr: '<OCR_WITH_REGION>',
  detect: '<OD>',
};

function quadToBox(q: number[]): [number, number, number, number] {
  const xs = [q[0], q[2], q[4], q[6]];
  const ys = [q[1], q[3], q[5], q[7]];
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)].map((v) => Math.round(v)) as [number, number, number, number];
}

export async function loadFlorence(m: Manifest, opts: CommonOptions) {
  return loadTjs(
    m,
    opts,
    async (tj, base) => {
      const { repo } = repoOf(m);
      const [model, processor] = await Promise.all([
        tj.Florence2ForConditionalGeneration.from_pretrained(repo, base as never),
        tj.AutoProcessor.from_pretrained(repo, { revision: base.revision, progress_callback: base.progress_callback as never }),
      ]);
      return { model, processor };
    },
    async (b) => {
      await (b.model as unknown as { dispose?: () => Promise<void> }).dispose?.();
    },
  );
}

export async function florence(
  m: Manifest,
  opts: CommonOptions,
  image: ImageLike,
  preset: FlorencePreset,
  maxTokens: number,
): Promise<{ result: FlorenceResult; info: RunInfo; inputTokens: number; outputTokens: number }> {
  const t = await getTransformers();
  const loaded = await loadFlorence(m, opts);
  const { model, processor } = loaded.value as unknown as {
    model: { generate: (o: unknown) => Promise<{ dims: number[] }> };
    processor: {
      construct_prompts: (t: string) => string[];
      (img: unknown, p: string[]): Promise<Record<string, unknown> & { input_ids: { dims: number[] } }>;
      batch_decode: (ids: unknown, o: unknown) => string[];
      post_process_generation: (text: string, task: string, size: [number, number]) => Record<string, unknown>;
    };
  };
  let task: string;
  let extra = '';
  if (typeof preset === 'object') {
    if (!preset.find) throw new ConfigError('preset { find } needs a phrase to look for.');
    task = '<CAPTION_TO_PHRASE_GROUNDING>';
    extra = preset.find;
  } else {
    task = TOKENS[preset];
    if (!task) throw new ConfigError(`Unknown preset "${preset}". Use caption, caption-detailed, caption-more-detailed, ocr, detect or { find }.`);
  }
  const raw = await toRawImage(t, image);
  const r = await serialize(`florence:${loaded.info.model}:${loaded.info.device}`, async () => {
    const prompts = processor.construct_prompts(task + extra);
    const inputs = await processor(raw, prompts);
    const ids = await model.generate({ ...inputs, max_new_tokens: maxTokens, num_beams: 1, do_sample: false });
    const text = processor.batch_decode(ids, { skip_special_tokens: false })[0];
    const post = processor.post_process_generation(text, task, [raw.width, raw.height]);
    return { post: post[task] as unknown, inputTokens: inputs.input_ids.dims.at(-1) ?? 0, outputTokens: ids.dims.at(-1) ?? 0 };
  });
  let result: FlorenceResult;
  const post = r.post;
  if (task === '<OD>' || task === '<CAPTION_TO_PHRASE_GROUNDING>') {
    const p = post as { bboxes: number[][]; labels: string[] };
    const boxes = (p.bboxes ?? []).map((b, i) => ({
      label: p.labels?.[i] || extra || 'object',
      box: b.map((v) => Math.round(v)) as [number, number, number, number],
    }));
    result = { text: boxes.map((b) => b.label).join(', '), object: { boxes } };
  } else if (task === '<OCR_WITH_REGION>') {
    const p = post as { quad_boxes: number[][]; labels: string[] };
    const regions = (p.labels ?? []).map((l, i) => ({ text: l.replace(/<\/?s>/g, '').trim(), box: quadToBox(p.quad_boxes[i]) }));
    result = { text: regions.map((x) => x.text).join('\n'), object: { regions } };
  } else {
    result = { text: String(post ?? '').trim() };
  }
  return { result, info: loaded.info, inputTokens: r.inputTokens, outputTokens: r.outputTokens };
}
