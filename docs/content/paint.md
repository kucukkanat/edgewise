---
title: paint
group: The six verbs
order: 50
eyebrow: verb · → image
---

# paint

Make an image from a prompt with SD-Turbo, a one-to-four-step distilled Stable Diffusion.

> [!NOTE] SD-Turbo is in preview: it is tested on Bun and Node, but not yet in a browser test run, because it needs a hardware GPU there. Pass `allowPreview: true`.

## Paint

```ts
import { paint } from 'edgewise';

const run = paint({ model: 'image:default', prompt: 'a lighthouse on a cliff at sunset, watercolor', size: '512x512', steps: 1, seed: 42, allowPreview: true });

for await (const step of run) preview.putImageData(new ImageData(step.preview.data, step.preview.width, step.preview.height), 0, 0);

const { image, seed, info } = await run;
image.width; image.height; image.data; // RGBA pixels
const blob = await image.toBlob();      // PNG
canvas.getContext('2d').putImageData(image.toImageData(), 0, 0); // browser
```

On servers, write `await image.toPng()` to a file.

## Options

| Option | Default | |
| --- | --- | --- |
| `prompt` | | what to draw |
| `size` | `'512x512'` | width and height from 256 to 1024, multiples of 64 |
| `steps` | 1 | 1 to 4; more steps add detail |
| `seed` | random | the same seed and prompt give the same image |

## Where it runs

SD-Turbo's weights are fp16 and about 2.5 GB. It needs WebGPU in the browser, ideally with `shader-f16`. On servers it runs on the CPU (a 256×256 image takes about 15 seconds on two cores) or on a GPU through vgpu. On the CPU Edgewise loads the text encoder, UNet and VAE one at a time to keep peak memory near 2 GB.

## Models

{{models-paint}}
