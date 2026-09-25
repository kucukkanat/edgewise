---
title: Message parts
group: Inputs
order: 1
---

# Message parts

Inputs are values, or parts that wrap them. Edgewise detects the type, checks it against the model's `accepts` list, and converts it for the model.

## What you can pass

| You pass | Becomes | Works in |
| --- | --- | --- |
| `string` | text | everywhere |
| `Blob` / `File` with an image type, `URL` of an image | image | everywhere (`URL` fetches) |
| `{ data, width, height, channels }` (`RawPixels`) | image | everywhere |
| `HTMLImageElement`, `HTMLCanvasElement`, `ImageBitmap`, `ImageData`, `OffscreenCanvas` | image | browser |
| `HTMLVideoElement` | image (current frame), or video with a video part | browser |
| `Blob` with an audio type, WAV bytes | audio | everywhere (other formats: browser) |
| `Float32Array` | audio at 16 kHz | everywhere |
| `mic()`, `audioSource()` | live audio | browser / everywhere |

## Explicit parts

```ts
await generate({
  model: 'vision:default',
  input: [
    { type: 'image', image: photo },
    { type: 'text', text: 'Describe this.' },
  ],
});

{ type: 'audio', audio: samples, sampleRate: 44100 } // resampled for you
{ type: 'video', video: videoEl, frames: 8 }         // browser: frames are sampled evenly
```

## Accepts

Every manifest lists its inputs:

```ts
registry.get('lfm2.5-vl-450m').accepts;  // ['text', 'image']
registry.list({ verb: 'generate', accepts: 'audio' }); // speech models
```

Passing an image to a text-only model throws `UnsupportedInputError` before any download, with a hint listing models that accept images.

## Audio decoding

WAV files decode in every runtime. In the browser, other formats (MP3, Ogg, WebM, M4A) decode with Web Audio. On servers, convert other formats to WAV or pass raw samples.
