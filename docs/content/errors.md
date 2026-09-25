---
title: Errors
group: Reference
order: 30
---

# Errors

Every error Edgewise throws is an `EdgewiseError` with a stable `code`, a `message`, and usually a `hint` that says what to do. `retryable` is true when trying again may work.

```ts
import { EdgewiseError, UnsupportedInputError } from 'edgewise';

try {
  await generate({ model: 'text:default', input: [photo, 'What is this?'] });
} catch (err) {
  if (err instanceof UnsupportedInputError) console.log(err.hint); // 'Models that accept image: lfm2.5-vl-450m, …'
  if (err instanceof EdgewiseError && err.retryable) retryLater();
}
```

| Class | Code | When |
| --- | --- | --- |
| `UnsupportedInputError` | `E_INPUT` | the model does not accept this input type, or a question type |
| `UnsupportedDeviceError` | `E_UNSUPPORTED` | the device cannot run the model (no WebGPU, browser-only API) |
| `ModelNotFoundError` | `E_MODEL` | unknown model ID or alias, or a preview model without `allowPreview` |
| `WrongVerbError` | `E_VERB` | a model used with the wrong verb, such as an embed model in `generate` |
| `DownloadError` | `E_DOWNLOAD` | a network failure, a bad status, or a SHA-256 mismatch (retryable) |
| `StorageQuotaError` | `E_QUOTA` | the browser refused to store the model |
| `OutOfMemoryError` | `E_OOM` | the GPU or process ran out of memory; try a smaller model or `dtype` |
| `BackendError` | `E_BACKEND` | ONNX Runtime failed to load or run the model |
| `SchemaValidationError` | `E_SCHEMA` | the output did not match the schema after a retry; has `raw` and `issues` |
| `PermissionError` | `E_PERMISSION` | the user blocked the microphone |
| `AbortError` | `E_ABORT` | the `signal` fired or `cancel()` was called |
| `ConfigError` | `E_CONFIG` | invalid options |

## Fallbacks

```ts
configure({
  fallback: {
    onUnsupported: 'next-variant', // or 'throw': try the next device instead of failing
    onBackendError: 'cpu',         // or 'throw': rerun on the CPU path after a GPU failure
  },
});
```

`info.device` in every result tells you whether a fallback happened. With `configure({ debug: true })` Edgewise logs each decision.
