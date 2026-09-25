---
title: Vercel AI SDK
group: Extras
order: 3
eyebrow: edgewise/ai-sdk
---

# Vercel AI SDK

`edgewise/ai-sdk` wraps Edgewise models as AI SDK providers (specification v4), so `streamText`, `generateText`, `embed` and tool calling in the `ai` package run on the device.

```bash
bun add edgewise ai @ai-sdk/provider
```

## Language models

```ts
import { streamText } from 'ai';
import { edgewise } from 'edgewise/ai-sdk';

const result = streamText({ model: edgewise('text:default'), prompt: 'Write a haiku about the sea.' });
for await (const delta of result.textStream) process.stdout.write(delta);
```

Images and audio in AI SDK file parts are passed to models that accept them. AI SDK tools work: Edgewise returns the tool calls and the AI SDK runs them.

## Embeddings

```ts
import { embedMany } from 'ai';
import { edgewiseEmbeddingModel } from 'edgewise/ai-sdk';

const { embeddings } = await embedMany({ model: edgewiseEmbeddingModel('embed:default', { dimensions: 256 }), values: docs });
```

## Evaluation models

`edgewiseEvaluationModel()` implements the evaluation model interface, so the same questions can run on a local encoder or, with `toAiSdkQuestions()`, on a cloud judge.

## Mixing local and cloud

```ts
import { openai } from '@ai-sdk/openai';

const model = (await capabilities()).tier === 'cpu' ? openai('gpt-5-mini') : edgewise('text:default');
```

`edgewise(id)` picks the right wrapper from the model's verb: generate, embed or evaluate.
