---
title: Transcription
group: Guides
order: 2
---

# Transcription

Speech recognition is `generate` with an audio input.

## A file

```ts
import { generate } from 'edgewise';

const { text } = await generate({ model: 'stt:tiny', input: file }); // WAV anywhere; MP3, Ogg, WebM in browsers
```

Or with the helper: `await transcribe(file)` from `edgewise/helpers` (uses `stt:realtime`).

## Timestamps and languages

Whisper returns segments and handles 99 languages. The Whisper models are in preview, so pass `allowPreview: true`:

```ts
const { segments, language } = await generate({
  model: 'stt:accurate',          // whisper-large-v3-turbo
  input: interview,
  timestamps: 'segment',
  language: 'auto',
  allowPreview: true,
});
// segments: [{ start: 0, end: 4.2, text: 'Welcome back…' }, …]
```

## Long audio

Audio over 30 seconds is cut at its pauses into windows of up to 28 seconds, with silent edges trimmed and silent windows skipped, then transcribed window by window and joined. This works the same for Moonshine and Whisper, so an hour-long recording needs no extra code. Timestamps are in seconds from the start of the file.

Follow progress with `onSegment`, which fires as each window finishes:

```ts
const { text, segments } = await generate({
  model: 'stt:multilingual',   // whisper-base
  input: recording,            // an MP3 Blob, an ArrayBuffer, or { type: 'audio', audio, sampleRate }
  timestamps: 'segment',
  allowPreview: true,
  onSegment: (segments, { done, total }) => {
    bar.value = done / total;
    for (const s of segments) list.append(`${s.start.toFixed(1)}s ${s.text}`);
  },
});
```

When you stream the run, each window also arrives as a `segment` event, and `signal` stops the run between windows. `splitOnSilence(samples)` from `edgewise/helpers` exposes the same splitter for 16 kHz audio.

## Live

```ts
import { generate, mic } from 'edgewise';

const run = generate({ model: 'stt:realtime', input: await mic({ vad: true }) });
for await (const delta of run) captions.textContent += delta;
```

See [Microphone and VAD](mic) for options and for `audioSource()` on servers.

## Which model

| Model | Alias | Size | Use for |
| --- | --- | --- | --- |
| `moonshine-tiny` | `stt:tiny` | ~30 MB | fast English, small devices |
| `moonshine-base` | `stt:realtime` | ~65 MB | live English captions |
| `whisper-tiny-en` | | ~40 MB | English, with timestamps |
| `whisper-base` | `stt:multilingual` | ~73 MB (CPU) | 99 languages, timestamps, small download |
| `whisper-small` | | ~240 MB (CPU) | 99 languages, more accurate than base |
| `whisper-large-v3-turbo` | `stt:accurate` | ~760 MB (CPU) | multilingual, timestamps |

Moonshine takes audio of any length without padding to 30 seconds, which is why it is faster for short utterances.
