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
await speak({ model: 'voice:default', speed: 1.2, input: 'A little faster.' }); // 0.5 to 2, Kokoro only
```

Kokoro has 28 English voices: American (`a…`) and British (`b…`), female (`?f_`) and male (`?m_`).

## Voice cloning

`chatterbox-turbo` (alias `voice:clone`) speaks in any voice from 3 to 10 seconds of reference audio. Cloning requires you to confirm the speaker agreed:

```ts
const reference = await fetch('/ana-sample.wav').then((r) => r.blob()); // one speaker, clear speech

await speak({
  model: 'voice:clone',
  allowPreview: true,
  voice: { reference, consent: { attested: true, by: 'Ana, signed release 2026-09-01' } },
  input: 'Hi, this is Ana. Well, a copy of her voice.',
}).play();
```

Encoding the reference takes a moment, so reuse it:

```ts
import { cloneVoice } from 'edgewise';

const ana = await cloneVoice({ reference, consent: { attested: true }, saveAs: 'ana', allowPreview: true });
await speak({ model: 'voice:clone', voice: ana, input: 'First line.', allowPreview: true });
await speak({ model: 'voice:clone', voice: 'saved:ana', input: 'Tomorrow, after a reload.', allowPreview: true });
```

`saveAs` keeps the voice (the speaker conditioning, not the recording) in the Origin Private File System in browsers and in the cache folder on servers. `exaggeration` (0 to 2, default 0.5) makes the delivery more or less expressive; `speed` is not supported.

Chatterbox Turbo is English only and downloads about 720 MB. Its 4-bit graphs need WebGPU in browsers; on servers it runs on the CPU (on two CPU cores, about four to five seconds of work per second of speech).

> [!RISK] A cloned voice can be used to impersonate someone. Clone only voices you have permission to use, tell listeners when speech is synthetic, and use `configure({ speak: { onSynthesize } })` to log what your app generates.

## Models

{{models-speak}}
