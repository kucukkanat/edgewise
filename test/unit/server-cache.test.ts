// Server-only: the download cache, shared downloads and cache.delete().
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { configure, resetConfig } from '../../src/core/config.ts';
import { AbortError } from '../../src/core/errors.ts';

const isServer = typeof window === 'undefined';

(isServer ? describe : describe.skip)('server download cache', () => {
  let dir: string;
  let base: string;
  let close: () => void;
  let hits = 0;
  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), 'ew-cache-'));
    configure({ cacheDir: dir });
    const body = Buffer.alloc(256 * 1024, 7);
    const server = createServer((req, res) => {
      hits++;
      res.writeHead(200, { 'content-length': String(body.length) });
      // Send slowly so a second caller can join the same download.
      let o = 0;
      const tick = () => {
        if (o >= body.length) return res.end();
        res.write(body.subarray(o, o + 16 * 1024));
        o += 16 * 1024;
        setTimeout(tick, 5);
      };
      tick();
    });
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', () => r()));
    const addr = server.address() as { port: number };
    base = `http://127.0.0.1:${addr.port}`;
    close = () => server.close();
  });
  afterAll(() => {
    close();
    rmSync(dir, { recursive: true, force: true });
    resetConfig();
  });

  it('shares one download, and one caller aborting does not fail the other', async () => {
    const { fetchToPath } = await import('../../src/platform/server.ts');
    hits = 0;
    const ac = new AbortController();
    let progressB = 0;
    const a = fetchToPath(`${base}/org/repo/resolve/abc/model.onnx`, { signal: ac.signal });
    const b = fetchToPath(`${base}/org/repo/resolve/abc/model.onnx`, { onProgress: () => progressB++ });
    setTimeout(() => ac.abort(), 10);
    await expect(a).rejects.toBeInstanceOf(AbortError);
    const path = await b;
    expect(path).toContain('org_repo_resolve_abc_model.onnx');
    expect(hits).toBe(1);
    expect(progressB).toBeGreaterThan(0);
  });

  it('rejects instead of crashing when the cache cannot be written', async () => {
    const { fetchToPath } = await import('../../src/platform/server.ts');
    writeFileSync(join(dir, 'blocker'), 'x');
    configure({ cacheDir: join(dir, 'blocker') }); // a file, so mkdir and writes fail
    await expect(fetchToPath(`${base}/x/y/resolve/z/other.bin`)).rejects.toThrow();
    configure({ cacheDir: dir });
  });

  it('cache.delete() matches repos and never removes saved voices or indexes', async () => {
    const { platform } = await import('../../src/platform/server.ts');
    await platform.store('edgewise-voice-keep.bin', new Uint8Array([1, 2, 3]));
    const n = await platform.cacheDelete('org/repo');
    expect(n).toBeGreaterThan(0);
    expect(await platform.store('edgewise-voice-keep.bin')).toEqual(new Uint8Array([1, 2, 3]));
    await platform.cacheDelete();
    expect(await platform.store('edgewise-voice-keep.bin')).toEqual(new Uint8Array([1, 2, 3]));
  });
});
