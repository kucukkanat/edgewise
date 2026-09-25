---
title: Build a voice assistant
group: Guides
order: 7
---

# Build a voice assistant

A private voice assistant that runs entirely in the browser: the microphone, speech recognition, a small language model with tools, and speech.

## The code

```ts file=assistant.ts
import { generate, mic, preload, speak, tool } from 'edgewise';
import { z } from 'zod';

await preload(['stt:realtime', 'text:default', 'voice:default', 'vad:default'], {
  onProgress: (e) => e.type === 'download' && (progress.value = e.loaded / e.total),
});

const tools = {
  setTimer: tool({
    description: 'Start a countdown timer',
    input: z.object({ minutes: z.number() }),
    execute: async ({ minutes }) => timers.start(minutes),
  }),
};

const messages = [];
const microphone = await mic({ vad: { redemptionMs: 700 } });

for await (const utterance of microphone.utterances()) {
  const { text } = await generate({ model: 'stt:realtime', input: utterance });
  if (!text.trim()) continue;
  messages.push({ role: 'user', content: text });

  const reply = generate({ model: 'text:default', system: 'You are a brief, friendly assistant.', messages, tools });
  await speak({ model: 'voice:default', input: reply }).play(); // starts with the first sentence
  messages.splice(0, messages.length, ...(await reply).messages);
}
```

## Budget

| Part | Model | Download |
| --- | --- | --- |
| Voice activity | Silero VAD | 2 MB |
| Speech to text | Moonshine base | ~65 MB |
| Language model | LFM2.5 350M | ~255 MB (WebGPU) |
| Text to speech | Kokoro 82M | ~90–330 MB |

About 400 to 650 MB in total, downloaded once. Because `speak` starts on the first complete sentence, the reply begins before the language model has finished writing it.
