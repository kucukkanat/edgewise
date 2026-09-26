/**
 * Edgewise: six verbs for on-device AI in the browser, Bun and Node.
 *
 * @packageDocumentation
 */
export { type ConfigureOptions, configure, type EdgewiseConfig, type FallbackPolicy, getConfig } from './core/config.ts';
export * from './core/errors.ts';
export { cache, persist, preload, storage } from './core/lifecycle.ts';
export type { AudioLike, AudioSource, ImageLike, Input, InputItem, Message, Part, RawPixels } from './core/parts.ts';
export { defineModel, type ListFilter, registry } from './core/registry.ts';
export { Run, type RunEvent } from './core/run.ts';
export { capabilities, loadedModels, unload } from './core/runtime.ts';
export type * from './core/types.ts';
export { audioSource, type Mic, type MicOptions, mic, type VadOptions } from './inputs/mic.ts';
export { type EmbedManyResult, type EmbedOptions, type EmbedResult, embed } from './verbs/embed.ts';
export { boolean, choice, type EvaluateOptions, type EvaluateResult, evaluate, label, type Question, type Span, score, spans } from './verbs/evaluate.ts';
export { type ForecastManyResult, type ForecastOptions, type ForecastResult, forecast, type SeriesForecast, type SeriesInput } from './verbs/forecast.ts';
export {
  type FlorencePreset,
  type GenerateChunk,
  type GenerateOptions,
  type GenerateResult,
  generate,
  type Segment,
  type ToolCall,
  type ToolResult,
} from './verbs/generate.ts';
export { type GeneratedImage, type PaintOptions, type PaintResult, type PaintStep, paint } from './verbs/paint.ts';
export type { SchemaLike } from './verbs/schema.ts';
export {
  type ClonedVoice,
  type CloneVoiceOptions,
  cloneVoice,
  listVoices,
  type SpeakOptions,
  type SpeechAudio,
  type SpeechChunk,
  speak,
  type Voice,
  type VoiceReference,
  type VoiceSpec,
} from './verbs/speak.ts';
export { type Tool, tool } from './verbs/tools.ts';
