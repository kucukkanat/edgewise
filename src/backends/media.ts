import { toEdgewiseError, UnsupportedInputError } from '../core/errors.ts';
import { type AudioLike, type ImageLike, isRawPixels } from '../core/parts.ts';
import { getPlatform } from '../core/runtime.ts';
import { resample } from '../platform/audio.ts';

type TJS = typeof import('@huggingface/transformers');
type RawImage = InstanceType<TJS['RawImage']>;

function is(v: unknown, ctor: string): boolean {
  const C = (globalThis as Record<string, unknown>)[ctor];
  return typeof C === 'function' && v instanceof (C as new (...a: never[]) => unknown);
}

function drawToPixels(src: CanvasImageSource, w: number, h: number): { data: Uint8ClampedArray; width: number; height: number } {
  const OC = (globalThis as { OffscreenCanvas?: typeof OffscreenCanvas }).OffscreenCanvas;
  if (OC) {
    const c = new OC(w, h);
    const ctx = c.getContext('2d') as OffscreenCanvasRenderingContext2D;
    ctx.drawImage(src, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  }
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d') as CanvasRenderingContext2D;
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/** Convert any supported image into a Transformers.js RawImage (RGB). */
export async function toRawImage(t: TJS, img: ImageLike): Promise<RawImage> {
  try {
    if (isRawPixels(img)) {
      return new t.RawImage(new Uint8ClampedArray(img.data), img.width, img.height, img.channels).rgb();
    }
    if (img instanceof URL) return (await t.RawImage.fromURL(img.href)).rgb();
    if (is(img, 'Blob')) return (await t.RawImage.fromBlob(img as Blob)).rgb();
    if (is(img, 'ImageData')) {
      const d = img as ImageData;
      return new t.RawImage(d.data, d.width, d.height, 4).rgb();
    }
    if (is(img, 'HTMLVideoElement')) {
      const v = img as HTMLVideoElement;
      if (!v.videoWidth) throw new UnsupportedInputError('The video element has no frame yet.', { hint: 'Wait for the video to start playing.' });
      const p = drawToPixels(v, v.videoWidth, v.videoHeight);
      return new t.RawImage(p.data, p.width, p.height, 4).rgb();
    }
    if (is(img, 'HTMLImageElement')) {
      const el = img as HTMLImageElement;
      if (!el.complete) await el.decode();
      const p = drawToPixels(el, el.naturalWidth, el.naturalHeight);
      return new t.RawImage(p.data, p.width, p.height, 4).rgb();
    }
    if (is(img, 'ImageBitmap')) {
      const b = img as ImageBitmap;
      const p = drawToPixels(b, b.width, b.height);
      return new t.RawImage(p.data, p.width, p.height, 4).rgb();
    }
    if (is(img, 'HTMLCanvasElement') || is(img, 'OffscreenCanvas')) {
      const c = img as HTMLCanvasElement;
      const p = drawToPixels(c, c.width, c.height);
      return new t.RawImage(p.data, p.width, p.height, 4).rgb();
    }
  } catch (err) {
    if (err instanceof UnsupportedInputError) throw err;
    throw toEdgewiseError(err, 'Reading the image');
  }
  throw new UnsupportedInputError('This value is not an image Edgewise can read.', {
    hint: 'Pass an image element, canvas, ImageBitmap, Blob, URL, or { data, width, height, channels }.',
  });
}

/** Convert any supported audio into mono Float32 samples at `rate` Hz (16 kHz by default). */
export async function toAudio(audio: AudioLike, sampleRate?: number, rate = 16000): Promise<Float32Array> {
  const p = getPlatform();
  if (audio instanceof Float32Array) return resample(audio, sampleRate ?? 16000, rate);
  if (is(audio, 'AudioBuffer')) {
    const b = audio as AudioBuffer;
    const out = new Float32Array(b.length);
    for (let c = 0; c < b.numberOfChannels; c++) {
      const d = b.getChannelData(c);
      for (let i = 0; i < d.length; i++) out[i] += d[i] / b.numberOfChannels;
    }
    return resample(out, b.sampleRate, rate);
  }
  let bytes: Uint8Array;
  let mime: string | undefined;
  if (audio instanceof URL) {
    const res = await fetch(audio.href);
    if (!res.ok) throw new UnsupportedInputError(`Could not fetch audio from ${audio.href} (HTTP ${res.status}).`);
    mime = res.headers.get('content-type') ?? undefined;
    bytes = new Uint8Array(await res.arrayBuffer());
  } else if (is(audio, 'Blob')) {
    mime = (audio as Blob).type || undefined;
    bytes = new Uint8Array(await (audio as Blob).arrayBuffer());
  } else if (audio instanceof ArrayBuffer) bytes = new Uint8Array(audio);
  else if (audio instanceof Uint8Array) bytes = audio;
  else throw new UnsupportedInputError('This value is not audio Edgewise can read.');
  const d = await p.decodeAudio(bytes, mime);
  return resample(d.samples, d.sampleRate, rate);
}
