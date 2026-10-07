import { connectWorker } from '../../src/worker/index.ts';

// Run as a child process by worker-terminate.test.ts: a panic here exits 133 instead of taking the test runner down.
const worker = new Worker(new URL('./ort.worker.ts', import.meta.url), { type: 'module' });
const loaded = new Promise((resolve) => worker.addEventListener('message', (e) => e.data === 'ort-loaded' && resolve(e.data)));
const ew = connectWorker(worker);
await loaded;
const pending = ew.models().then(
  () => 'resolved',
  (err: Error) => err.name,
);
await ew.terminate();
// Give a crashing teardown time to surface before reporting success.
await Bun.sleep(500);
console.log(`terminated, pending run: ${await pending}`);
