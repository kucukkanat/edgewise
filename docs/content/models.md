---
title: Models
group: Reference
order: 10
description: Every model Edgewise supports, with sizes, devices, status and licence, generated from the registry.
---

# Models

This table is generated from the registry in the package, so it matches the version you install. Filter by verb or input. Rows you cannot run in this browser are dimmed.

{{models-table}}

## Status

- **stable**: real-model tests pass on Bun, Node and Chromium. Available by default.
- **preview**: works, but is tested on fewer runtimes or has known limits. Needs `allowPreview: true`.
- **experimental**: new or unusual hosting. Needs `allowPreview: true`, and may change.

## Aliases

Aliases pick a good default for a job. They may point to a better model in a minor release; use the full ID to pin one.

{{alias-table}}

## Where models come from

- Most models are ONNX exports on Hugging Face, pinned to a commit.
- Models with no official ONNX export (Chronos-Bolt and the LiquidAI LFM2.5 encoders) were exported by Edgewise and are served from the [`kucukkanat/edgewise-models`](https://huggingface.co/buckets/kucukkanat/edgewise-models) Hugging Face bucket. Each file is checked against a SHA-256 hash on download.

The export scripts are in `scripts/export` in the repository.

## Licences

Each model keeps its own licence, shown in the table. Restrict what your app can load:

```ts
configure({ licenses: ['apache-2.0', 'mit'] });
```
