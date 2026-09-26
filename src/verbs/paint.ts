import { latentPreview, sdTurbo } from '../backends/sdturbo.ts';
import { ConfigError } from '../core/errors.ts';
import { encodePng } from '../core/png.ts';
import { resolveManifest } from '../core/registry.ts';
import { createRun, type Run } from '../core/run.ts';
import type { CommonOptions, ModelRef, RunInfo } from '../core/types.ts';

export interface PaintOptions extends CommonOptions {
  model: ModelRef;
  prompt: string;
  /** `'512x512'` (default). Width and height must be multiples of 64. */
  size?: `${number}x${number}`;
  /** 1 to 4. Default 1. */
  steps?: number;
  /** Same seed and prompt give the same image. Random by default. */
  seed?: number;
}

export interface GeneratedImage {
  width: number;
  height: number;
  /** RGBA pixels. */
  data: Uint8ClampedArray;
  /** Encode as PNG (or JPEG / WebP where the platform supports it). */
  toBlob(type?: 'image/png' | 'image/jpeg' | 'image/webp'): Promise<Blob>;
  /** PNG bytes. Works in every runtime; write them to a file on servers. */
  toPng(): Promise<Uint8Array>;
  /** Browser only: an ImageData you can put on a canvas. */
  toImageData(): ImageData;
}

export interface PaintResult {
  image: GeneratedImage;
  seed: number;
  info: RunInfo;
}

export interface PaintStep {
  step: number;
  total: number;
  /** A low-resolution RGBA preview decoded directly from latents. */
  preview: { data: Uint8ClampedArray; width: number; height: number };
}

/** Wrap RGBA pixels as a GeneratedImage (used when results cross a worker boundary). */
export function imageFromRgba(rgba: Uint8ClampedArray, width: number, height: number): GeneratedImage {
  const rgb = new Uint8ClampedArray(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    rgb[i * 3] = rgba[i * 4];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return makeImage(rgb, width, height);
}

function makeImage(rgb: Uint8ClampedArray, width: number, height: number): GeneratedImage {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4] = rgb[i * 3];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  return {
    width,
    height,
    data: rgba,
    async toBlob(type = 'image/png') {
      const OC = (globalThis as { OffscreenCanvas?: typeof OffscreenCanvas }).OffscreenCanvas;
      if (OC) {
        const c = new OC(width, height);
        (c.getContext('2d') as OffscreenCanvasRenderingContext2D).putImageData(new ImageData(rgba, width, height), 0, 0);
        return c.convertToBlob({ type });
      }
      if (type !== 'image/png') throw new ConfigError(`${type} needs OffscreenCanvas, which this runtime lacks.`, { hint: "Use 'image/png' or toPng()." });
      return new Blob([(await encodePng(rgba, width, height)) as Uint8Array<ArrayBuffer>], { type: 'image/png' });
    },
    toPng: () => encodePng(rgba, width, height),
    toImageData() {
      if (typeof ImageData === 'undefined') throw new ConfigError('ImageData only exists in browsers. Use image.data instead.');
      return new ImageData(rgba, width, height);
    },
  };
}

/** Generate an image from a prompt. Iterate the run for a preview after every step. */
export function paint(options: PaintOptions): Run<PaintResult, PaintStep> {
  return createRun<PaintResult, PaintStep>(async (ctx) => {
    if (!options.prompt?.trim()) throw new ConfigError('paint() needs a prompt.');
    const m = resolveManifest(options.model, { verb: 'paint', allowPreview: options.allowPreview, inputs: ['text'] });
    const [w, h] = (options.size ?? '512x512').split('x').map(Number);
    if (!(w >= 256 && h >= 256 && w <= 1024 && h <= 1024 && w % 64 === 0 && h % 64 === 0)) {
      throw new ConfigError(`size must be between 256 and 1024 and a multiple of 64, got "${options.size}".`);
    }
    const seed = options.seed ?? Math.floor(Math.random() * 2 ** 31);
    const r = await sdTurbo(
      m,
      {
        ...options,
        signal: ctx.signal,
        onProgress: (e) => {
          options.onProgress?.(e);
          ctx.event({ type: 'load', event: e });
        },
      },
      {
        prompt: options.prompt,
        width: w,
        height: h,
        steps: options.steps ?? 1,
        seed,
        signal: ctx.signal,
        onStep: (step, total, lat, [lh, lw]) => {
          ctx.event({ type: 'step', step, total });
          if (lh === lw) ctx.emit({ step, total, preview: latentPreview(lat, lh) });
          else ctx.emit({ step, total, preview: { data: new Uint8ClampedArray(lw * lh * 4), width: lw, height: lh } });
        },
      },
    );
    return { image: makeImage(r.rgb, r.width, r.height), seed, info: r.info };
  }, options.signal);
}
