import { InferenceSession } from 'onnxruntime-node';
import { serveWorker } from '../../src/worker/index.ts';

serveWorker();
// Loads onnxruntime-node's N-API binding without downloading a model: creating a session from a non-ONNX file
// still constructs (and later finalizes) the native session wrapper, which is what Bun < 1.4 panics on.
await InferenceSession.create(new URL('../../package.json', import.meta.url).pathname).catch(() => undefined);
postMessage('ort-loaded');
