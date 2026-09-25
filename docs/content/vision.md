---
title: Images, video and documents
group: Guides
order: 1
---

# Images, video and documents

## Ask about an image

```ts
import { generate } from 'edgewise';

const { text } = await generate({ model: 'vision:default', input: [imageFile, 'What is written on the sign?'] });
```

`vision:default` is LFM2.5-VL-450M: about 675 MB (920 MB on WebAssembly), and it runs on WebGPU, WASM and CPU. `lfm2.5-vl-1.6b` is more capable and needs about 1.8 GB. `smolvlm-256m` is the smallest.

## Captions, OCR and boxes

Florence-2 is faster than a chat VLM for fixed tasks. The helpers wrap its presets:

```ts
import { caption, detect, ocr } from 'edgewise/helpers';

const { text } = await caption(photo, { detail: 'detailed' });
const read = await ocr(receipt);            // read.text, read.object.regions: [{ text, box }]
const found = await detect(photo);          // found.object.boxes: [{ label, box }]
const cars = await detect(photo, 'a red car'); // phrase grounding
```

Boxes are `[x1, y1, x2, y2]` in pixels of the original image.

## Fill a schema from a photo

```ts
import { z } from 'zod';

const { object } = await generate({
  model: 'vision:default',
  input: [receiptPhoto, 'Extract the receipt.'],
  schema: z.object({ store: z.string(), total: z.number(), date: z.string() }),
});
```

## Video

In the browser, a video part samples frames evenly and sends them as images:

```ts
await generate({ model: 'vision:default', input: [{ type: 'video', video: videoEl, frames: 6 }, 'What happens in this clip?'] });
```

## Watch a webcam

```ts
import { watch } from 'edgewise/helpers';

const stop = watch(videoEl, {
  model: 'vision:default',
  prompt: 'Is anyone at the door? Answer yes or no.',
  everyMs: 2000,
  onResult: ({ text }) => (status.textContent = text),
});
// later: stop()
```

`watch` skips a frame while the previous one is still being answered, so it never queues up.
