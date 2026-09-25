import { debug } from '../core/config.ts';
import { capabilities, getPlatform } from '../core/runtime.ts';

/**
 * Dot products between one query and many vectors, on the GPU through vgpu when it pays off,
 * otherwise on the CPU. Rows are stored back to back in `matrix`.
 */

type Gpu = { device: unknown; dispose(): void };
type Storage = { write(d: ArrayBufferView, off?: number): void; read(): Promise<ArrayBuffer>; destroy(): void; size: number };
type Kernel = { set(v: Record<string, unknown>): Kernel; dispatch(x: number): void };
interface Vgpu {
  init(o?: Record<string, unknown>): Promise<Gpu>;
  storage(g: Gpu, bytes: number, access: 'read' | 'read-write'): Storage;
  compute(g: Gpu, src: string): Kernel;
}

const WGSL = `
struct Params { rows: u32, dims: u32 };
@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<storage, read> params: Params;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= params.rows) { return; }
  var s = 0.0;
  let base = row * params.dims;
  for (var d = 0u; d < params.dims; d++) { s += matrix[base + d] * query[d]; }
  scores[row] = s;
}`;

let session: Promise<{ v: Vgpu; gpu: Gpu; kernel: Kernel }> | null = null;
let idle: ReturnType<typeof setTimeout> | null = null;
const resident = new WeakMap<Float32Array, { buf: Storage; length: number }>();

function scheduleRelease(): void {
  if (idle) clearTimeout(idle);
  idle = setTimeout(async () => {
    const s = await session?.catch(() => null);
    session = null;
    s?.gpu.dispose();
    debug('vgpu released after idle');
  }, 5000);
  (idle as { unref?: () => void }).unref?.();
}

async function acquire(): Promise<{ v: Vgpu; gpu: Gpu; kernel: Kernel }> {
  session ??= (async () => {
    const browser = getPlatform().isBrowser;
    const v = (browser ? await import('vgpu') : await import('vgpu/node')) as unknown as Vgpu;
    const caps = await capabilities();
    const gpu = await v.init(browser ? {} : { adapter: caps.hardwareGpu ? 'hardware' : 'software' });
    return { v, gpu, kernel: v.compute(gpu, WGSL) };
  })();
  session.catch(() => {
    session = null;
  });
  return session;
}

export function cpuScores(matrix: Float32Array, dims: number, query: Float32Array): Float32Array {
  const rows = Math.floor(matrix.length / dims);
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    const base = r * dims;
    for (let d = 0; d < dims; d++) s += matrix[base + d] * query[d];
    out[r] = s;
  }
  return out;
}

export async function gpuScores(matrix: Float32Array, dims: number, query: Float32Array): Promise<Float32Array> {
  const { v, gpu, kernel } = await acquire();
  const rows = Math.floor(matrix.length / dims);
  let res = resident.get(matrix);
  if (!res || res.length !== matrix.length) {
    res?.buf.destroy();
    const buf = v.storage(gpu, matrix.byteLength, 'read');
    buf.write(matrix);
    res = { buf, length: matrix.length };
    resident.set(matrix, res);
  }
  const q = v.storage(gpu, query.byteLength, 'read');
  q.write(query);
  const out = v.storage(gpu, rows * 4, 'read-write');
  const params = v.storage(gpu, 8, 'read');
  params.write(new Uint32Array([rows, dims]));
  kernel.set({ matrix: res.buf, query: q, scores: out, params }).dispatch(Math.ceil(rows / 64));
  const scores = new Float32Array(await out.read());
  q.destroy();
  out.destroy();
  params.destroy();
  scheduleRelease();
  return scores;
}

export type GpuMode = 'auto' | 'on' | 'off';

/** Scores on the GPU for large indexes on devices with a hardware GPU, otherwise on the CPU. */
export async function scores(
  matrix: Float32Array,
  dims: number,
  query: Float32Array,
  mode: GpuMode = 'auto',
): Promise<{ scores: Float32Array; device: 'gpu' | 'cpu' }> {
  if (mode === 'off') return { scores: cpuScores(matrix, dims, query), device: 'cpu' };
  const big = matrix.length >= 4_000_000;
  if (mode === 'on' || big) {
    const caps = await capabilities();
    if (mode === 'on' || (caps.webgpu && caps.hardwareGpu)) {
      try {
        return { scores: await gpuScores(matrix, dims, query), device: 'gpu' };
      } catch (err) {
        if (mode === 'on') throw err;
        debug('GPU scoring failed, using CPU:', err instanceof Error ? err.message : err);
      }
    }
  }
  return { scores: cpuScores(matrix, dims, query), device: 'cpu' };
}
