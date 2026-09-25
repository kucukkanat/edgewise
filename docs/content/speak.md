---
title: speak
group: The six verbs
order: 40
eyebrow: verb · → audio
---

# speak

Turn text, or a live stream of text, into speech.

## Speak and play

```ts
import { speak } from 'edgewise';

// browser: start playing with the first sentence
await speak({ model: 'voice:default', voice: 'af_heart', input: 'Hello from your device.' }).play();

// anywhere: get the audio
const audio = await speak({ model: 'voice:default', input: 'Hello.' });
audio.samples;    // Float32Array, mono
audio.sampleRate; // 24000
audio.duration;   // seconds
audio.toWav();    // Uint8Array, 16-bit PCM WAV
```

`speak` needs the optional `phonemizer` package for Kokoro. It bundles espeak-ng, which is GPL-3.0; check that it suits your app.

## Stream text in

`input` can be any async iterable of strings, such as a running `generate()`. Speech is made one sentence at a time, as soon as each sentence is complete:

```ts
const reply = generate({ model: 'text:default', input: question });
const run = speak({ model: 'voice:default', input: reply });

for await (const chunk of run) console.log(chunk.text, chunk.samples.length);
```

`.play()` schedules those chunks back to back without gaps.

## Voices

```ts
import { listVoices } from 'edgewise';

const voices = await listVoices('voice:default'); // [{ id: 'af_heart', name: 'Heart', lang: 'en-US', gender: 'female', grade: 'A' }, …]

await speak({ model: 'voice:default', voice: 'bm_george', input: 'Good evening.' });
await speak({ model: 'voice:default', voice: { af_heart: 0.7, af_bella: 0.3 }, input: 'A blended voice.' }); // weighted blend
await speak({ model: 'voice:default', speed: 1.2, input: 'A little faster.' }); // 0.5 to 2
```

Kokoro has 28 English voices: American (`a…`) and British (`b…`), female (`?f_`) and male (`?m_`).

## Voice cloning

No model in the registry clones voices yet. Passing `voice: { reference, consent }` throws `UnsupportedInputError` today. The API is reserved so that apps written now keep working when a cloning model is added; see the [roadmap](roadmap).

## Models

{{models-speak}}
