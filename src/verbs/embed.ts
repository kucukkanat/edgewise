import { embedTexts } from '../backends/embedding.ts';
import { mockEmbed } from '../backends/mock.ts';
import { ConfigError, UnsupportedInputError } from '../core/errors.ts';
import { resolveManifest } from '../core/registry.ts';
import type { CommonOptions, ModelRef, RunInfo } from '../core/types.ts';

interface EmbedBase extends CommonOptions {
  model: ModelRef;
  /** Some models embed queries and documents differently. Edgewise adds the right prefix. */
  purpose?: 'query' | 'document';
  /** Shorten vectors (models trained for it, such as EmbeddingGemma: 768, 512, 256, 128). */
  dimensions?: number;
  /** Scale vectors to unit length. Default true. */
  normalize?: boolean;
}

export interface EmbedOneOptions extends EmbedBase {
  input: string;
  values?: never;
}
export interface EmbedManyOptions extends EmbedBase {
  values: string[];
  input?: never;
  /** Called after each batch. */
  onBatch?: (progress: { done: number; total: number }) => void;
}
export type EmbedOptions = EmbedOneOptions | EmbedManyOptions;

export interface EmbedResult {
  embedding: Float32Array;
  model: string;
  dimensions: number;
  info: RunInfo;
}
export interface EmbedManyResult {
  embeddings: Float32Array[];
  model: string;
  dimensions: number;
  info: RunInfo;
}

/** Turn text into vectors that sit close together when meanings are close. */
export async function embed(options: EmbedOneOptions): Promise<EmbedResult>;
export async function embed(options: EmbedManyOptions): Promise<EmbedManyResult>;
export async function embed(options: EmbedOptions): Promise<EmbedResult | EmbedManyResult> {
  const many = options.values !== undefined;
  if (many === (options.input !== undefined)) throw new ConfigError('Pass either input (one value) or values (many).');
  const items = many ? (options.values as string[]) : [options.input as string];
  if (items.some((x) => typeof x !== 'string')) {
    throw new UnsupportedInputError('embed() currently accepts text only.', { hint: 'Pass strings in input or values.' });
  }
  const m = resolveManifest(options.model, { verb: 'embed', allowPreview: options.allowPreview, inputs: ['text'] });
  const native = (m.config?.dimensions as number | undefined) ?? undefined;
  const allowed = m.config?.matryoshka as number[] | undefined;
  if (options.dimensions !== undefined) {
    if (!allowed) throw new ConfigError(`"${m.id}" cannot shorten its vectors.`, { hint: `"${m.id}" always returns ${native ?? 'its native'} dimensions.` });
    if (!allowed.includes(options.dimensions)) throw new ConfigError(`"${m.id}" supports dimensions ${allowed.join(', ')}.`);
  }
  const prefixes = m.config?.prefixes as { query?: string; document?: string } | undefined;
  const prefix = options.purpose && prefixes ? (prefixes[options.purpose] ?? '') : '';
  const texts = items.map((t) => prefix + t);
  const { vectors, info } =
    m.task === 'mock'
      ? mockEmbed(m, texts)
      : await embedTexts(m, options, texts, {
          dimensions: options.dimensions,
          normalize: options.normalize,
          onBatch: many && (options as EmbedManyOptions).onBatch ? (done, total) => (options as EmbedManyOptions).onBatch?.({ done, total }) : undefined,
        });
  const dimensions = vectors[0]?.length ?? options.dimensions ?? native ?? 0;
  const tag = options.dimensions ? `${m.id}@${options.dimensions}` : m.id;
  if (many) return { embeddings: vectors, model: tag, dimensions, info };
  return { embedding: vectors[0], model: tag, dimensions, info };
}
