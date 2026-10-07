// Regression for https://github.com/oven-sh/bun/issues/30286: terminating a worker that loaded onnxruntime-node
// used to panic Bun (< 1.4) with "NAPI FATAL ERROR: Error::New napi_create_error" (exit 133).
describe('terminate() on Bun', () => {
  it.skipIf(typeof Bun === 'undefined')(
    'stops a worker that loaded onnxruntime-node without crashing the process',
    async () => {
      const proc = Bun.spawn([process.execPath, '--conditions=source', new URL('../fixtures/terminate-ort-worker.ts', import.meta.url).pathname], {
        stdout: 'pipe',
        stderr: 'pipe',
      });
      const [code, out, err] = await Promise.all([proc.exited, new Response(proc.stdout).text(), new Response(proc.stderr).text()]);
      expect({ code, out: out.trim(), panic: err.includes('panic') }).toEqual({ code: 0, out: 'terminated, pending run: AbortError', panic: false });
    },
    30_000,
  );
});
