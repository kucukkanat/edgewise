import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    'helpers/index': 'src/helpers/index.ts',
    'react/index': 'src/react/index.ts',
    'ai-sdk/index': 'src/ai-sdk/index.ts',
    'test/index': 'src/test/index.ts',
    'worker/index': 'src/worker/index.ts',
    'platform/browser': 'src/platform/browser.ts',
    'platform/server': 'src/platform/server.ts',
  },
  format: ['esm'],
  target: 'es2022',
  platform: 'neutral',
  splitting: true,
  sourcemap: true,
  clean: true,
  dts: false,
  treeshake: true,
  // Keep `node:` specifiers so Deno and edge bundlers resolve builtins.
  removeNodeProtocol: false,
  external: [
    '#platform',
    '@huggingface/transformers',
    'onnxruntime-web',
    'onnxruntime-web/webgpu',
    'onnxruntime-node',
    'vgpu',
    'vgpu/node',
    'phonemizer',
    'react',
    'zod',
    'ai',
    '@ai-sdk/provider',
    /^node:/,
  ],
});
