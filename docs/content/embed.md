---
title: embed
group: The six verbs
order: 30
eyebrow: verb · → vectors
---

# embed

Turn text into vectors that sit close together when meanings are close. Use them for semantic search, retrieval for RAG, deduplication and clustering.

## One or many

```ts
import { embed } from 'edgewise';

const { embedding, dimensions } = await embed({ model: 'embed:default', input: 'How do I reset my password?' });

const { embeddings } = await embed({
  model: 'embed:tiny',
  values: docs.map((d) => d.text),
  onBatch: ({ done, total }) => (bar.value = done / total),
});
```

`input` takes one string and returns `embedding`. `values` takes many and returns `embeddings`, in batches of 16. Vectors are `Float32Array`, normalized to unit length unless you pass `normalize: false`, so a dot product is the cosine similarity.

## Queries and documents

EmbeddingGemma embeds search queries and documents with different prefixes. Say which one you are embedding and Edgewise adds the right prefix:

```ts
await embed({ model: 'embeddinggemma-300m', values: chunks, purpose: 'document' });
await embed({ model: 'embeddinggemma-300m', input: question, purpose: 'query' });
```

Models without prefixes ignore `purpose`.

## Shorter vectors

EmbeddingGemma is trained so its first dimensions carry the most meaning. Pass `dimensions` (768, 512, 256 or 128) to trade a little accuracy for a smaller index:

```ts
const r = await embed({ model: 'embeddinggemma-300m', input: 'hello', dimensions: 256 });
r.model; // 'embeddinggemma-300m@256'
```

`result.model` records the model and size. Store it with your vectors: vectors from different models, or different sizes, cannot be compared.

## Similarity

```ts
import { cosine } from 'edgewise/helpers';
cosine(a, b); // -1..1
```

For search over many vectors use [`vectorIndex()`](search).

## Models

{{models-embed}}
