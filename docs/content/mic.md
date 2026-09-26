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

`m.speaking` is true from the moment speech starts until its utterance ends, and `m.speechProbability` is the latest VAD probability. For events, pass callbacks:

| Callback | When |
| --- | --- |
| `onSpeechStart()` | the moment speech starts, long before the utterance ends |
| `onSpeechEnd(audio)` | an utterance ends (the same audio `utterances()` yields) |
| `onMisfire()` | speech started but was shorter than `minSpeechMs` |
| `onFrame(p)` | every 32 ms, with the speech probability |

Callbacks fire while something reads `utterances()`.

## Interruptions (barge-in)

Cancel the reply when the user starts talking. Cancelling a `generate()` stops it mid-token; cancelling a `speak()` stops playback at once, and `play()` then rejects with `AbortError`:

```ts
let turn: AbortController | null = null;
const m = await mic({ vad: { onSpeechStart: () => turn?.abort() } });

for await (const heard of m.utterances()) {
  turn = new AbortController();
  const { signal } = turn;
  const { text } = await generate({ model: 'stt:tiny', input: heard, signal });
  const reply = generate({ model: 'text:default', input: text, signal });
  speak({ model: 'voice:default', input: reply, signal }).play().catch(() => {}); // not awaited: keep listening
}
```

On laptop speakers the assistant's own voice can reach the microphone. Two guards help: interrupt only if speech lasts a little while (check `m.speaking` again after 200 to 300 ms), and ignore transcripts that mostly repeat what the assistant just said. With headphones, interrupt at once.

## Noise and echo

The browser's echo cancellation, noise suppression and automatic gain control are on by default. Turn them off with `mic({ echoCancellation: false, noiseSuppression: false, autoGainControl: false })`, for example with headphones or for music. In noisy rooms raise `positiveThreshold` and `minSpeechMs` so short sounds do not count as speech.

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
