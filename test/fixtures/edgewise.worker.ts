import { serveWorker } from '../../src/worker/index.ts';
import { defineWorkerMocks } from './mocks.ts';

defineWorkerMocks();
serveWorker();
