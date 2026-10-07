import { type D1Question, d1Decide, pyJson } from '../backends/d1.ts';
import { classifyLabels, nliChoice, type RawSpan, tokenSpans } from '../backends/judge.ts';
import { lfmRoute, lfmSpans } from '../backends/lfm.ts';
import { toAudio, toRawImage } from '../backends/media.ts';
import { mockEvaluate } from '../backends/mock.ts';
import { getTransformers } from '../backends/transformers.ts';
import { ConfigError, UnsupportedInputError } from '../core/errors.ts';
import type { AudioLike, ImageLike } from '../core/parts.ts';
import { resolveManifest } from '../core/registry.ts';
import { checkSignal } from '../core/runtime.ts';
import type { CommonOptions, Manifest, ModelRef, PartType, RunInfo } from '../core/types.ts';
import { confidenceOf } from '../core/util.ts';

interface QBase {
  /** Model for this question. Defaults to the call's `model`. */
  model?: ModelRef;
  threshold?: number;
  /**
   * What to judge, in words, such as "Which team should handle this ticket?".
   * Decision models like d1 and cloud judges like Jev read it; encoder models score the option descriptions directly.
   */
  instructions?: string;
}

export interface ChoiceQuestion<K extends string = string> extends QBase {
  kind: 'choice';
  options: Record<K, string>;
  /** Returned as the choice when the top probability is below `threshold`. */
  otherwise?: K;
}
export interface ScoreQuestion<L extends string = string> extends QBase {
  kind: 'score';
  levels: readonly L[];
}
export interface BooleanQuestion extends QBase {
  kind: 'boolean';
  true?: string;
  false?: string;
}
export interface SpansQuestion extends QBase {
  kind: 'spans';
  types?: string[];
}
export interface LabelQuestion extends QBase {
  kind: 'label';
}

export type Question = ChoiceQuestion<string> | ScoreQuestion<string> | BooleanQuestion | SpansQuestion | LabelQuestion;

/** Pick one of up to 255 named options. The answer's `choice` is typed as your keys. */
export function choice<K extends string>(options: Record<K, string>, opts: Omit<ChoiceQuestion<NoInfer<K>>, 'kind' | 'options'> = {}): ChoiceQuestion<K> {
  const keys = Object.keys(options);
  if (keys.length < 2) throw new ConfigError('choice() needs at least two options.');
  if (keys.length > 255) throw new ConfigError('choice() supports at most 255 options.');
  if (opts.otherwise !== undefined && !(opts.otherwise in options)) throw new ConfigError(`otherwise "${opts.otherwise}" is not one of the options.`);
  return { kind: 'choice', options, ...opts };
}

/** Rate on an ordered rubric of 2 to 10 levels. `score` is the probability-weighted mean level index. */
export function score<L extends string>(levels: readonly L[], opts: Omit<ScoreQuestion<L>, 'kind' | 'levels'> = {}): ScoreQuestion<L> {
  if (levels.length < 2 || levels.length > 10) throw new ConfigError('score() needs 2 to 10 levels.');
  return { kind: 'score', levels, ...opts };
}

/** Probability that a statement is true. Pass descriptions for true and false with zero-shot models. */
export function boolean(opts: Omit<BooleanQuestion, 'kind'> = {}): BooleanQuestion {
  return { kind: 'boolean', ...opts };
}

/** Find spans of the given entity types (all types when omitted). */
export function spans(types?: string[], opts: Omit<SpansQuestion, 'kind' | 'types'> = {}): SpansQuestion {
  return { kind: 'spans', types, ...opts };
}

/** The label of a fixed-label classifier. */
export function label(opts: Omit<LabelQuestion, 'kind'> = {}): LabelQuestion {
  return { kind: 'label', ...opts };
}

export interface Span {
  type: string;
  start: number;
  end: number;
  text: string;
  score: number;
}

export type Answer<Q> =
  Q extends ChoiceQuestion<infer K>
    ? { choice: K; probabilities: Record<K, number>; fellBack: boolean; flagged?: boolean }
    : Q extends ScoreQuestion<infer L>
      ? { score: number; level: L; probabilities: Record<L, number>; flagged?: boolean }
      : Q extends BooleanQuestion
        ? { probability: number; flagged?: boolean }
        : Q extends SpansQuestion
          ? { spans: Span[]; flagged?: boolean }
          : Q extends LabelQuestion
            ? { label: string; probabilities: Record<string, number>; flagged?: boolean }
            : never;

export interface EvaluateOptions<Q extends Record<string, Question>> extends CommonOptions {
  /** Default model for questions that do not name one. */
  model?: ModelRef;
  /**
   * What to judge: a string, or JSON (serialized with sorted keys; d1 models read it as JSON).
   * Optional when images or audio are the whole state.
   */
  state?: unknown;
  questions: Q;
  /** Images to judge along with the state, in order. Needs a model that accepts images, such as d1-omni-600m. */
  images?: ImageLike | ImageLike[];
  /** One audio clip (up to 30 s) to judge along with the state. Needs a model that accepts audio. */
  audio?: AudioLike;
  /** Sample rate of `audio` when it is a Float32Array at a rate other than 16 kHz. */
  sampleRate?: number;
}

export interface EvaluateResult<Q extends Record<string, Question>> {
  answers: { [K in keyof Q]: Answer<Q[K]> };
  /** 1 minus normalized entropy per question: 1 when certain, 0 when spread evenly. */
  confidence: { [K in keyof Q]: number };
  /** Which model and device answered each question. */
  info: { [K in keyof Q]: RunInfo };
}

/** Serialize state deterministically: strings as-is, objects as sorted `key: value` lines. */
export function stateToText(state: unknown): string {
  if (typeof state === 'string') return state;
  if (state === null || state === undefined) throw new ConfigError('evaluate() needs a state.');
  if (typeof state !== 'object') return String(state);
  if (Array.isArray(state)) return state.map(stateToText).join('\n');
  const sorted = (v: unknown): unknown =>
    v && typeof v === 'object' && !Array.isArray(v)
      ? Object.fromEntries(
          Object.keys(v as object)
            .sort()
            .map((k) => [k, sorted((v as Record<string, unknown>)[k])]),
        )
      : Array.isArray(v)
        ? v.map(sorted)
        : v;
  return Object.entries(sorted(state) as Record<string, unknown>)
    .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join('\n');
}

function matchesType(type: string, wanted?: string[]): boolean {
  if (!wanted?.length) return true;
  const t = type.toUpperCase().replace(/[^A-Z]/g, '');
  return wanted.some((w) => {
    const x = w.toUpperCase().replace(/[^A-Z]/g, '');
    return t === x || t.includes(x) || x.includes(t);
  });
}

/**
 * Ask typed questions about text or JSON and get probabilities back from small encoder models.
 * Questions naming different models run on those models; each question is scored independently.
 */
export async function evaluate<Q extends Record<string, Question>>(options: EvaluateOptions<Q>): Promise<EvaluateResult<Q>> {
  if (!Object.keys(options.questions ?? {}).length) throw new ConfigError('evaluate() needs at least one question.');
  const images = options.images === undefined ? [] : Array.isArray(options.images) ? options.images : [options.images];
  const media: PartType[] = [...(images.length ? (['image'] as const) : []), ...(options.audio !== undefined ? (['audio'] as const) : [])];
  const hasState = options.state !== undefined && options.state !== null;
  if (!hasState && !media.length) throw new ConfigError('evaluate() needs a state, images or audio.');
  let text: string | undefined;
  const stateText = () => {
    text ??= stateToText(options.state);
    return text;
  };
  const answers: Record<string, unknown> = {};
  const confidence: Record<string, number> = {};
  const info: Record<string, RunInfo> = {};
  const common: CommonOptions = {
    signal: options.signal,
    onProgress: options.onProgress,
    device: options.device,
    dtype: options.dtype,
    allowPreview: options.allowPreview,
  };
  const resolved: [string, Question, Manifest][] = Object.entries(options.questions).map(([name, q]) => {
    const ref = q.model ?? options.model;
    if (!ref) throw new ConfigError(`Question "${name}" has no model. Pass model on evaluate() or on the question.`);
    return [name, q, resolveManifest(ref, { verb: 'evaluate', allowPreview: options.allowPreview, inputs: ['text', ...media] })];
  });
  // d1 answers all of a model's questions in one pass, sharing the encoded media.
  const decided = new Map<string, Result>();
  const groups = new Map<string, [string, Question, Manifest][]>();
  for (const r of resolved) if (r[2].task === 'd1-decision') groups.set(r[2].id, [...(groups.get(r[2].id) ?? []), r]);
  const t = groups.size && images.length ? await getTransformers() : undefined;
  const raw = t ? await Promise.all(images.map((i) => toRawImage(t, i))) : undefined;
  const audio = groups.size && options.audio !== undefined ? await toAudio(options.audio, options.sampleRate) : undefined;
  // As d1 was trained: JSON as JSON, and no state is "" with images and {} with audio.
  const state = typeof options.state === 'string' ? options.state : hasState ? pyJson(options.state) : audio ? '{}' : '';
  for (const group of groups.values()) {
    checkSignal(options.signal);
    const m = group[0][2];
    const { probs, info: i } = await d1Decide(m, common, { state, questions: group.map(([n, q]) => toD1(m, n, q)), images: raw, audio });
    group.forEach(([name, q], k) => decided.set(name, fromProbs(q, probs[k], i)));
  }
  for (const [name, q, m] of resolved) {
    checkSignal(options.signal);
    const r = decided.get(name) ?? (m.task === 'mock' ? mockAnswer(m, name, q, stateText()) : await answer(m, q, stateText(), common));
    const a = r.answer as { flagged?: boolean } & Record<string, unknown>;
    if (q.threshold !== undefined) a.flagged = flag(q, a, q.threshold);
    answers[name] = a;
    confidence[name] = r.confidence;
    info[name] = r.info;
  }
  return { answers, confidence, info } as EvaluateResult<Q>;
}

type Result = { answer: unknown; confidence: number; info: RunInfo };

/** Map an Edgewise question onto d1's schema. d1 reads instructions, so each kind has a neutral default. */
function toD1(m: Manifest, name: string, q: Question): D1Question {
  if (q.kind === 'choice') {
    return { type: 'choice', instructions: q.instructions ?? 'Which option fits best?', options: Object.entries(q.options) };
  }
  if (q.kind === 'score') return { type: 'score', instructions: q.instructions ?? 'Which level fits best?', levels: q.levels };
  if (q.kind === 'boolean') {
    const instructions = q.instructions ?? q.true;
    if (!instructions) {
      throw new ConfigError(`boolean() question "${name}" on "${m.id}" needs instructions or a description of what true means.`, {
        hint: "Pass boolean({ instructions: 'Is the customer asking for a refund?' }).",
      });
    }
    return { type: 'noul', instructions, true: q.true, false: q.false };
  }
  throw new UnsupportedInputError(`"${m.id}" cannot answer ${q.kind}() questions.`, {
    hint: `"${m.id}" answers: ${(m.features ?? []).join(', ')}. Pass a different model on this question.`,
  });
}

/** An answer from a distribution over the question's options (a boolean's as [false, true]). */
function fromProbs(q: Question, probs: number[], info: RunInfo): Result {
  if (q.kind === 'boolean') return { answer: { probability: probs[1] }, confidence: confidenceOf(probs), info };
  if (q.kind === 'score') {
    const mean = probs.reduce((s, p, i) => s + p * i, 0);
    return {
      answer: { score: mean / (probs.length - 1), level: q.levels[Math.round(mean)], probabilities: Object.fromEntries(q.levels.map((l, i) => [l, probs[i]])) },
      confidence: confidenceOf(probs),
      info,
    };
  }
  if (q.kind !== 'choice') throw new UnsupportedInputError(`${q.kind}() questions have no option distribution.`);
  const keys = Object.keys(q.options);
  let best = 0;
  probs.forEach((p, i) => {
    if (p > probs[best]) best = i;
  });
  const fellBack = q.threshold !== undefined && probs[best] < q.threshold && q.otherwise !== undefined;
  return {
    answer: { choice: fellBack ? q.otherwise : keys[best], probabilities: Object.fromEntries(keys.map((k, i) => [k, probs[i]])), fellBack },
    confidence: confidenceOf(probs),
    info,
  };
}

function mockAnswer(m: Manifest, name: string, q: Question, text: string): Result {
  const { value, info } = mockEvaluate(m, name, text);
  if (value === undefined) throw new ConfigError(`The mock model returned nothing for question "${name}".`);
  if (q.kind === 'boolean') {
    const p = typeof value === 'number' ? value : Number((value as Record<string, number>).true ?? 0);
    return { answer: { probability: p }, confidence: confidenceOf([p, 1 - p]), info };
  }
  if (q.kind === 'spans') {
    const spans = (value as { type: string; start: number; end: number; score?: number }[]).map((s) => ({
      ...s,
      score: s.score ?? 1,
      text: text.slice(s.start, s.end),
    }));
    return { answer: { spans: spans.filter((s) => matchesType(s.type, q.types)) }, confidence: 1, info };
  }
  const probs = value as Record<string, number>;
  const keys = q.kind === 'choice' ? Object.keys(q.options) : q.kind === 'score' ? [...q.levels] : Object.keys(probs);
  const ps = keys.map((k) => probs[k] ?? 0);
  const sum = ps.reduce((a, b) => a + b, 0) || 1;
  const norm = ps.map((p) => p / sum);
  let best = 0;
  norm.forEach((p, i) => {
    if (p > norm[best]) best = i;
  });
  const probabilities = Object.fromEntries(keys.map((k, i) => [k, norm[i]]));
  if (q.kind === 'choice') {
    const fellBack = q.threshold !== undefined && norm[best] < q.threshold && q.otherwise !== undefined;
    return { answer: { choice: fellBack ? q.otherwise : keys[best], probabilities, fellBack }, confidence: confidenceOf(norm), info };
  }
  if (q.kind === 'score') {
    // The expected level index, scaled to 0..1 as documented.
    const mean = norm.reduce((s, p, i) => s + p * i, 0);
    return { answer: { score: mean / (norm.length - 1), level: q.levels[Math.round(mean)], probabilities }, confidence: confidenceOf(norm), info };
  }
  return { answer: { label: keys[best], probabilities }, confidence: confidenceOf(norm), info };
}

function flag(q: Question, a: Record<string, unknown>, th: number): boolean {
  switch (q.kind) {
    case 'boolean':
      return (a.probability as number) >= th;
    case 'score':
      return (a.score as number) >= th;
    case 'spans':
      return (a.spans as Span[]).some((s) => s.score >= th);
    case 'choice':
      return a.fellBack as boolean;
    case 'label':
      return Math.max(...Object.values(a.probabilities as Record<string, number>)) < th;
  }
}

async function answer(m: Manifest, q: Question, text: string, o: CommonOptions): Promise<{ answer: unknown; confidence: number; info: RunInfo }> {
  const unsupported = () =>
    new UnsupportedInputError(`"${m.id}" cannot answer ${q.kind}() questions.`, {
      hint: `"${m.id}" answers: ${(m.features ?? []).join(', ') || 'none'}. Pass a different model on this question.`,
    });
  const lanes = async (descs: string[], single = false) => {
    if (m.task === 'lfm-router') {
      const r = await lfmRoute(m, o, text, single ? [descs[0], 'Anything else'] : descs);
      return { probs: single ? [r.probs[0]] : r.probs, info: r.info };
    }
    return nliChoice(m, o, text, descs, single);
  };
  if (m.task === 'zero-shot-nli' || m.task === 'lfm-router') {
    if (q.kind === 'choice') {
      const keys = Object.keys(q.options);
      const { probs, info } = await lanes(keys.map((k) => q.options[k] || k));
      let best = 0;
      probs.forEach((p, i) => {
        if (p > probs[best]) best = i;
      });
      const fellBack = q.threshold !== undefined && probs[best] < q.threshold && q.otherwise !== undefined;
      return {
        answer: { choice: fellBack ? q.otherwise : keys[best], probabilities: Object.fromEntries(keys.map((k, i) => [k, probs[i]])), fellBack },
        confidence: confidenceOf(probs),
        info,
      };
    }
    if (q.kind === 'score') {
      const { probs, info } = await lanes([...q.levels]);
      const mean = probs.reduce((s, p, i) => s + p * i, 0);
      return {
        answer: {
          score: mean / (probs.length - 1),
          level: q.levels[Math.round(mean)],
          probabilities: Object.fromEntries(q.levels.map((l, i) => [l, probs[i]])),
        },
        confidence: confidenceOf(probs),
        info,
      };
    }
    if (q.kind === 'boolean') {
      if (!q.true) {
        throw new ConfigError(`boolean() on "${m.id}" needs a description of what true means.`, {
          hint: "Pass boolean({ true: 'The customer asks for a refund' }), or use a model with a boolean head.",
        });
      }
      if (q.false) {
        const { probs, info } = await lanes([q.true, q.false]);
        return { answer: { probability: probs[0] }, confidence: confidenceOf(probs), info };
      }
      const { probs, info } = await lanes([q.true], true);
      return { answer: { probability: probs[0] }, confidence: confidenceOf([probs[0], 1 - probs[0]]), info };
    }
    throw unsupported();
  }
  if (m.task === 'sequence-classification') {
    const { labels, probs, info } = await classifyLabels(m, o, text);
    if (q.kind === 'boolean') {
      const pos = (m.config?.positiveLabel as string | undefined) ?? labels[labels.length - 1];
      const i = labels.indexOf(pos);
      const p = i >= 0 ? probs[i] : probs[probs.length - 1];
      return { answer: { probability: p }, confidence: confidenceOf([p, 1 - p]), info };
    }
    if (q.kind === 'label') {
      let best = 0;
      probs.forEach((p, i) => {
        if (p > probs[best]) best = i;
      });
      return { answer: { label: labels[best], probabilities: Object.fromEntries(labels.map((l, i) => [l, probs[i]])) }, confidence: confidenceOf(probs), info };
    }
    if (q.kind === 'choice') {
      const keys = Object.keys(q.options);
      const idx = keys.map((k) => labels.findIndex((l) => l.toLowerCase() === k.toLowerCase()));
      if (idx.some((i) => i < 0)) throw new ConfigError(`choice() keys on "${m.id}" must be its labels: ${labels.join(', ')}.`);
      const ps = idx.map((i) => probs[i]);
      const s = ps.reduce((a, b) => a + b, 0) || 1;
      const norm = ps.map((p) => p / s);
      let best = 0;
      norm.forEach((p, i) => {
        if (p > norm[best]) best = i;
      });
      const fellBack = q.threshold !== undefined && norm[best] < q.threshold && q.otherwise !== undefined;
      return {
        answer: { choice: fellBack ? q.otherwise : keys[best], probabilities: Object.fromEntries(keys.map((k, i) => [k, norm[i]])), fellBack },
        confidence: confidenceOf(norm),
        info,
      };
    }
    throw unsupported();
  }
  if (m.task === 'token-classification' || m.task === 'lfm-token-classification') {
    if (q.kind !== 'spans') throw unsupported();
    const { spans: raw, info } = m.task === 'lfm-token-classification' ? await lfmSpans(m, o, text) : await tokenSpans(m, o, text);
    const out: Span[] = raw.filter((s: RawSpan) => matchesType(s.type, q.types));
    const top = out.length ? Math.max(...out.map((s) => s.score)) : 1;
    return { answer: { spans: out }, confidence: top, info };
  }
  throw unsupported();
}
