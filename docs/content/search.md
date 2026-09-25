---
title: Semantic search and RAG
group: Guides
order: 5
---

# Semantic search and RAG

## Build an index

```ts
import { embed } from 'edgewise';
import { vectorIndex } from 'edgewise/helpers';

const index = await vectorIndex<{ title: string }>({ name: 'help-articles', model: 'embed:default' });

if (index.size === 0) {
  const { embeddings } = await embed({ model: 'embed:default', values: articles.map((a) => a.body), purpose: 'document' });
  await index.add(embeddings.map((vector, i) => ({ id: articles[i].id, vector, meta: { title: articles[i].title } })));
  await index.save();
}
```

`vectorIndex` is an exact nearest-neighbour index over normalized vectors, fast up to about 100,000 vectors. `save()` writes it to the Origin Private File System in browsers and to `<cache>/indexes` on servers. The next `vectorIndex()` with the same name loads it, as long as the model matches.

## Query

```ts
const hits = await index.query('how do I change my email?', { k: 5, minScore: 0.3 });
// [{ id: 'a42', score: 0.71, meta: { title: 'Update your account email' } }, …]
```

`query` embeds the text with the index's model and `purpose: 'query'`. Use `search(vector)` when you already have a vector.

## Retrieval-augmented generation

```ts
import { generate } from 'edgewise';

const hits = await index.query(question, { k: 4 });
const context = hits.map((h) => articles.find((a) => a.id === h.id)!.body).join('\n---\n');

const { text } = await generate({
  model: 'text:default',
  system: 'Answer only from the context. Say "I don\'t know" if the answer is not there.',
  input: `Context:\n${context}\n\nQuestion: ${question}`,
});
```

## Find duplicates

```ts
const pairs = await index.pairs({ minScore: 0.92 }); // [{ a, b, score }]
```

## GPU scoring

Scoring runs on the CPU for small indexes. With a hardware GPU (WebGPU in the browser, vgpu on servers) and more than about four million floats, Edgewise scores on the GPU with a WGSL kernel. Force it either way with `vectorIndex({ …, gpu: 'on' | 'off' })`.
