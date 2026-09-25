/**
 * A tiny forwarding server for browser model tests.
 * Headless Chromium in some sandboxes cannot reach the internet directly, so the browser tests
 * point `hub` at this server, which fetches (through any HTTPS_PROXY) and caches on disk.
 *
 *   bun test/models/hub-proxy.ts            # listens on :8787
 *   EDGEWISE_TEST_HUB=http://localhost:8787 vitest run --project browser-models
 */
import { createHash } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, renameSync } from 'node:fs';
import { join } from 'node:path';

const port = Number(process.env.PORT ?? 8787);
const dir = process.env.HUB_PROXY_CACHE ?? join(process.cwd(), '.cache', 'hub-proxy');
mkdirSync(dir, { recursive: true });
const upstream: Record<string, string> = { hf: 'https://huggingface.co', cdn: 'https://cdn.jsdelivr.net' };
const cors = { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-expose-headers': '*' };
const inflight = new Map<string, Promise<boolean>>();

async function fill(url: string, file: string): Promise<boolean> {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) return false;
  // Bun.write(path, response) can stall on proxied responses; stream by hand.
  const out = createWriteStream(`${file}.part`);
  for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) if (!out.write(chunk)) await new Promise((r) => out.once('drain', r));
  await new Promise<void>((r, j) => out.end((e?: Error) => (e ? j(e) : r())));
  renameSync(`${file}.part`, file);
  return true;
}

Bun.serve({
  port,
  idleTimeout: 255,
  async fetch(req: Request) {
    if (req.method === 'OPTIONS') return new Response(null, { headers: cors });
    const u = new URL(req.url);
    const [, head, ...rest] = u.pathname.split('/');
    const base = upstream[head];
    if (!base) return new Response('unknown upstream', { status: 404, headers: cors });
    const target = `${base}/${rest.join('/')}${u.search}`;
    const file = join(dir, createHash('sha256').update(target).digest('hex').slice(0, 32));
    if (!existsSync(file)) {
      let p = inflight.get(file);
      if (!p) {
        p = fill(target, file).finally(() => inflight.delete(file));
        inflight.set(file, p);
      }
      if (!(await p)) return new Response('upstream error', { status: 404, headers: cors });
    }
    const f = Bun.file(file);
    const size = f.size;
    const type = /\.m?js$/.test(u.pathname)
      ? 'text/javascript'
      : u.pathname.endsWith('.wasm')
        ? 'application/wasm'
        : u.pathname.endsWith('.json')
          ? 'application/json'
          : 'application/octet-stream';
    const headers: Record<string, string> = { ...cors, 'accept-ranges': 'bytes', 'content-type': type };
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.get('range') ?? '');
    if (range) {
      const start = range[1] ? Number(range[1]) : Math.max(0, size - Number(range[2]));
      const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
      headers['content-range'] = `bytes ${start}-${end}/${size}`;
      headers['content-length'] = String(end - start + 1);
      return new Response(req.method === 'HEAD' ? null : f.slice(start, end + 1), { status: 206, headers });
    }
    headers['content-length'] = String(size);
    return new Response(req.method === 'HEAD' ? null : f, { headers });
  },
});
console.log(`hub proxy on http://localhost:${port} (cache ${dir})`);
