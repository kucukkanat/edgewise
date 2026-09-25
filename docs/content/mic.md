---
title: Microphone and VAD
group: Inputs
order: 2
---

# Microphone and VAD

`mic()` records from the user's microphone in the browser. With voice activity detection (Silero VAD) it splits speech into utterances.

## Record

```ts
import { generate, mic } from 'edgewise';

const m = await mic();          // asks for permission
await m.start();
// …
const samples = await m.stop(); // Float32Array at 16 kHz
const { text } = await generate({ model: 'stt:tiny', input: samples });
m.dispose();                    // release the microphone
```

`m.level` is the current input level (0 to 1), for a meter. `m.toBlob()` records a compressed copy with MediaRecorder.

## Utterances

```ts
const m = await mic({ vad: true });
for await (const utterance of m.utterances()) {
  // one Float32Array per thing the user said, trimmed of silence
}
```

VAD options: `positiveThreshold` (0.5), `negativeThreshold` (0.35), `minSpeechMs` (250), `redemptionMs` (600, the silence that ends an utterance) and `preSpeechPadMs` (300).

## Live transcription

Pass the microphone straight to `generate`. Each utterance is transcribed when it ends:

```ts
const run = generate({ model: 'stt:realtime', input: await mic({ vad: true }) });
for await (const delta of run) captions.textContent += delta;
```

For whole utterances, read the events instead:

```ts
for await (const e of run.events) if (e.type === 'utterance-end') log(e.text);
```

The Run keeps going until the source ends (`m.dispose()`) or you cancel it.

## Servers: audioSource()

On servers, wrap any stream of PCM chunks:

```ts
import { audioSource, generate } from 'edgewise';

const source = audioSource(pcmChunks, { sampleRate: 48000, vad: true }); // AsyncIterable<Float32Array>
for await (const delta of generate({ model: 'stt:tiny', input: source })) process.stdout.write(delta);
```

## Find speech in a recording

```ts
import { detectSpeech } from 'edgewise/helpers';
const segments = await detectSpeech(wavBlob); // [{ start: 1.02, end: 3.4, audio }, …] in seconds
```

## Permissions

`mic()` throws `PermissionError` when the user blocks the microphone, and `UnsupportedDeviceError` outside the browser. Browsers only allow microphone access on HTTPS and localhost.
