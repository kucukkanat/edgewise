---
title: generate
group: The six verbs
order: 10
eyebrow: verb · → text
---

# generate

One function for every model that writes text: tiny language models, vision-language models, tool callers, speech recognizers and Florence-2.

## Text in, text out

```ts
import { generate } from 'edgewise';

const { text, usage, finishReason, info } = await generate({
  model: 'lfm2.5-350m',
  system: 'You are terse.',
  input: 'Three uses for a small language model inside a web page?',
  maxTokens: 200,
});
```

`temperature` defaults to 0, which decodes greedily and gives repeatable output. Set it above 0, with `topP`, to sample.

## Any input the model accepts

The call is the same whatever the input. Only the model has to accept it.

```ts
// image + question → answer
await generate({ model: 'vision:default', input: [photo, 'What is the person holding?'] });

// audio → transcript
await generate({ model: 'stt:tiny', input: wavBlob });

// image → OCR, using a Florence-2 preset
await generate({ model: 'florence-2-base', input: screenshot, preset: 'ocr' });

// short video → summary (browser; frames are sampled for you)
await generate({ model: 'vision:default', input: [{ type: 'video', video: videoEl, frames: 8 }, 'Summarize.'] });
```

Passing an input the model does not accept throws `UnsupportedInputError` before anything downloads, with a hint naming models that do accept it.

## Conversations

```ts
const first = await generate({
  model: 'vision:default',
  messages: [{ role: 'user', content: [{ type: 'image', image: chartPng }, { type: 'text', text: 'What does this show?' }] }],
});
const next = await generate({ model: 'vision:default', messages: [...first.messages, { role: 'user', content: 'Which month was highest?' }] });
```

Use `input` for single turns and `messages` for conversations. They cannot be combined. `result.messages` is the conversation including the reply.

## Streaming

```ts
const run = generate({ model: 'text:default', input: prompt, signal: controller.signal });
for await (const delta of run) output.textContent += delta;
const { text, usage } = await run;
```

Special tokens and tool-call markup are filtered out of the stream.

## Structured output

Pass a Zod schema (or any object with `safeParse`) and read `object`. With a schema, the stream yields partial objects instead of text.

```ts
import { z } from 'zod';

const { object } = await generate({
  model: 'text:default',
  input: 'Lunch with Ana on 2026-10-02 at Foodhallen',
  schema: z.object({ who: z.string(), date: z.string(), place: z.string().optional() }),
});
// { who: 'Ana', date: '2026-10-02', place: 'Foodhallen' }
```

Edgewise sends the JSON Schema to the model, extracts the JSON from the reply, and validates it. If validation fails it retries once with the errors. If that fails too it throws `SchemaValidationError` with the raw text and the issues.

## Tools

```ts
import { generate, tool } from 'edgewise';
import { z } from 'zod';

const { text, toolCalls } = await generate({
  model: 'lfm2.5-350m',
  input: 'Set the volume to 30.',
  tools: {
    setVolume: tool({
      description: 'Set the speaker volume from 0 to 100',
      input: z.object({ level: z.number().int() }),
      execute: async ({ level }) => player.setVolume(level),
    }),
  },
});
```

See [Tool calling](tools) for approval, multi-step loops and model formats.

## Presets

Florence-2 is trained on fixed prompts. `preset` picks one and fills `object`.

| Preset | Result |
| --- | --- |
| `'caption'`, `'caption-detailed'`, `'caption-more-detailed'` | `text` |
| `'ocr'` | `text`, `object.regions` with quadrilaterals |
| `'detect'` | `object.boxes` with labels |
| `{ find: 'a red car' }` | `object.boxes` for the phrase |

## Speech models

Audio input goes to speech recognizers the same way. `timestamps: 'segment'` returns `segments` (Whisper). `language` sets the spoken language for multilingual models. With a live `mic()` or `audioSource()`, `generate` transcribes each utterance as it ends. See [Transcription](transcription).

## Chrome's built-in model

In Chrome with the Prompt API, `'chrome:gemini-nano'` runs Gemini Nano with no download for your app. Pass a list to prefer it and fall back:

```ts
await generate({ model: ['chrome:gemini-nano', 'text:default'], input: 'Summarize: …' });
```

## Options

| Option | Default | |
| --- | --- | --- |
| `model` | | ID, alias, manifest, or a list tried in order |
| `input` / `messages` | | one of the two |
| `system` | | system prompt |
| `schema` | | fill a Zod schema |
| `tools`, `maxSteps`, `approve` | `maxSteps: 3` | tool calling |
| `preset` | | Florence-2 tasks |
| `timestamps`, `language` | | speech models |
| `maxTokens` | 512 | per step |
| `temperature`, `topP`, `stop` | 0 | decoding |
| `device`, `dtype`, `signal`, `onProgress`, `allowPreview` | | shared by every verb |

## Which model

{{models-generate}}
