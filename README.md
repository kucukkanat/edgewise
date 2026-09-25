# Edgewise

**Six verbs for on-device AI in TypeScript.** Edgewise runs language, vision, speech, judging, embedding, image and forecasting models where your code runs: in the browser through WebGPU or WebAssembly, and in Bun or Node through ONNX Runtime, with [vgpu](https://github.com/vercel/vgpu) for WebGPU on the server.

[Documentation](https://kucukkanat.github.io/edgewise/) · [Live demos](https://kucukkanat.github.io/edgewise/demos.html) · [Models](https://kucukkanat.github.io/edgewise/models.html) · [API reference](https://kucukkanat.github.io/edgewise/api/)

```bash
bun add edgewise
```

```ts
import { generate, evaluate, embed, speak, paint, forecast, choice } from 'edgewise';

// generate → text, from text, images or audio
const { text } = await generate({ model: 'text:default', input: 'Three names for a bakery?' });

// evaluate → typed decisions from small encoders
const { answers } = await evaluate({
  model: 'judge:router',
  state: 'My invoice charged me twice.',
  questions: { lane: choice({ billing: 'billing', tech: 'technical support', chat: 'small talk' }) },
});

// embed → vectors
const { embedding } = await embed({ model: 'embed:default', input: 'How do I reset my password?' });

// speak → audio (streams sentence by sentence)
const audio = await speak({ model: 'voice:default', input: 'Hello from your device.' });

// paint → image (preview)
const { image } = await paint({ model: 'image:default', prompt: 'a lighthouse at sunset, watercolor', allowPreview: true });

// forecast → numbers with ranges
const { median, quantiles } = await forecast({ model: 'forecast:default', series: hourly, horizon: 24 });
```

## Why

- **Six verbs, grouped by output.** A vision-language model and a speech recognizer both produce text, so both use `generate`. New model families plug into a verb instead of adding API.
- **One package, three runtimes.** The same code runs in browsers, Bun and Node. Every result reports which device and backend actually ran.
- **Await or stream, same call.** `generate`, `speak` and `paint` return a Run: `await` it, or `for await` it.
- **Pinned, verified models.** Every model is pinned to an exact revision; files Edgewise hosts are checked against SHA-256.
- **Tested with real models.** A model is `stable` only when real-model tests pass on Bun, Node and Chromium.

## Entry points

| Import | |
| --- | --- |
| `edgewise` | the verbs, `mic()`, `audioSource()`, `tool()`, question builders, `configure()`, `capabilities()`, `registry` |
| `edgewise/helpers` | `route`, `lint`, `redact`, `caption`, `ocr`, `vectorIndex`, `detectAnomalies`, … |
| `edgewise/react` | hooks: `useModel`, `useChat`, `useEvaluate`, `useSpeak`, … |
| `edgewise/ai-sdk` | Vercel AI SDK providers |
| `edgewise/test` | mock models for unit tests |

Optional peers: `zod` (schemas, tools), `phonemizer` (Kokoro TTS; bundles GPL-3.0 espeak-ng), `react`, `ai` and `@ai-sdk/provider`.

## Development

```bash
bun install
bun run typecheck && bun run lint
bun run test            # unit tests on Bun, Node and Chromium
bun run test:models     # real models on all three (downloads ~5 GB)
bun run build           # dist/
bun run docs          # docs/dist, the GitHub Pages site
```

The ONNX export scripts for the models Edgewise hosts (Chronos-Bolt, the LiquidAI LFM2.5 encoders) are in `scripts/export`.

## Licence

MIT. Each model keeps its own licence, listed in the [model table](https://kucukkanat.github.io/edgewise/models.html).
