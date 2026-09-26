/** Shared setup for real-model tests. They download models, so they only run when asked. */
import { configure, unload } from '../../src/index.ts';

declare const __EDGEWISE_TEST_HUB__: string | undefined;
/** Must match the onnxruntime-web version Transformers.js depends on. */
const ORT_WEB = '1.31.0-dev.20260914-8d85527a0';

const env: Record<string, string | undefined> = (globalThis as { process?: { env: Record<string, string> } }).process?.env ?? {};
export const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';
export const runtime = isBrowser ? 'browser' : (globalThis as { Bun?: unknown }).Bun ? 'bun' : 'node';
export const enabled = isBrowser || env.EDGEWISE_MODEL_TESTS === '1';

const hub = typeof __EDGEWISE_TEST_HUB__ === 'string' && __EDGEWISE_TEST_HUB__ ? __EDGEWISE_TEST_HUB__ : undefined;
/** The configuration model tests use, also applied inside workers. */
export const testConfig = {
  allowPreview: true,
  maxLoadedModels: 1,
  ...(env.EDGEWISE_CACHE ? { cacheDir: env.EDGEWISE_CACHE } : {}),
  ...(hub ? { hub: `${hub}/hf`, wasmPaths: `${hub}/cdn/npm/onnxruntime-web@${ORT_WEB}/dist/` } : {}),
};
// One model at a time keeps the browser tab and small CI runners within memory.
configure(testConfig);

/** `describe` when model tests are on, `describe.skip` otherwise. Unloads models afterwards to keep memory flat. */
export function suite(name: string, fn: () => void) {
  if (!enabled) return describe.skip(name, fn);
  return describe(name, () => {
    afterAll(async () => {
      await unload();
    });
    fn();
  });
}

/** Skip a test on the listed runtimes (with a reason shown in the name). */
export function on(runtimes: string[], name: string): [string, boolean] {
  return [runtimes.includes(runtime) ? name : `${name} (not on ${runtime})`, !runtimes.includes(runtime)];
}
export const it2 = (spec: [string, boolean], fn: () => Promise<void>, timeout = 1_800_000) => (spec[1] ? it.skip(spec[0], fn) : it(spec[0], fn, timeout));

/** A simple test image: a red disc on white. */
export function redDisc(size = 224): { data: Uint8ClampedArray; width: number; height: number; channels: 3 } {
  const data = new Uint8ClampedArray(size * size * 3);
  const c = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 3;
      const inside = (x - c) ** 2 + (y - c) ** 2 < (size * 0.35) ** 2;
      data[i] = 255;
      data[i + 1] = inside ? 0 : 255;
      data[i + 2] = inside ? 0 : 255;
    }
  }
  return { data, width: size, height: size, channels: 3 };
}

/** Log the backend that ran, so CI output shows what each runtime used. */
export function report(name: string, info: unknown) {
  console.log(`[${runtime}] ${name}`, JSON.stringify(info));
}
