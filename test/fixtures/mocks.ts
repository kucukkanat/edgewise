// Mock models shared by the worker tests. Imported both in-process and inside a real worker.
import { mockModel } from '../../src/test/index.ts';

export function defineWorkerMocks() {
  mockModel({ id: 'wk:echo', verb: 'generate', respond: ({ messages }) => `echo: ${String(messages.at(-1)?.content)}` });
  mockModel({
    id: 'wk:tools',
    verb: 'generate',
    respond: ({ messages }) =>
      messages.some((m) => m.role === 'tool') ? `done: ${String(messages.at(-1)?.content)}` : { toolCalls: [{ name: 'add', input: { a: 2, b: 3 } }] },
  });
  mockModel({ id: 'wk:json', verb: 'generate', respond: () => '{"name":"Ana","age":"not a number"}' });
  mockModel({ id: 'wk:judge', verb: 'evaluate', respond: () => ({ lane: { billing: 0.8, other: 0.2 } }) });
  // Answers with a label describing the media that reached it, so tests can check what crossed the bridge.
  mockModel({
    id: 'wk:media',
    verb: 'evaluate',
    respond: ({ state, images, audio, sampleRate }) => {
      const img = images.map((i) => ('width' in i && 'channels' in i ? `px${i.width}x${i.height}` : (i as object).constructor.name)).join('+');
      const pcm = audio instanceof Float32Array ? `pcm${audio.length}@${sampleRate ?? 16000}` : audio === undefined ? '' : (audio as object).constructor.name;
      return { seen: { [`${state}|${img}|${pcm}`]: 1 } };
    },
  });
  mockModel({ id: 'wk:embed', verb: 'embed', dimensions: 3, respond: ({ texts }) => texts.map((t) => [t.length, 1, 0]) });
  mockModel({ id: 'wk:voice', verb: 'speak' });
  mockModel({ id: 'wk:fc', verb: 'forecast', respond: ({ horizon }) => Array.from({ length: horizon }, (_, i) => i) });
  mockModel({ id: 'wk:slow', verb: 'generate', respond: async () => (await new Promise((r) => setTimeout(r, 200)), 'late') });
}
