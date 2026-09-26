/**
 * Every error Edgewise throws extends {@link EdgewiseError}. Each carries a
 * stable `code`, a `retryable` flag and a `hint` written for end users.
 */
export type ErrorCode =
  | 'E_INPUT'
  | 'E_UNSUPPORTED'
  | 'E_MODEL'
  | 'E_VERB'
  | 'E_DOWNLOAD'
  | 'E_QUOTA'
  | 'E_OOM'
  | 'E_BACKEND'
  | 'E_SCHEMA'
  | 'E_PERMISSION'
  | 'E_ABORT'
  | 'E_CONFIG';

export interface EdgewiseErrorOptions {
  hint?: string;
  retryable?: boolean;
  cause?: unknown;
}

export class EdgewiseError extends Error {
  readonly code: ErrorCode;
  readonly hint: string;
  readonly retryable: boolean;

  constructor(code: ErrorCode, message: string, options: EdgewiseErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = new.target.name;
    this.code = code;
    this.hint = options.hint ?? message;
    this.retryable = options.retryable ?? false;
  }
}

/** A part type was passed to a model that does not accept it. */
export class UnsupportedInputError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_INPUT', message, options);
  }
}

/** The device or runtime cannot run this model or feature. */
export class UnsupportedDeviceError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_UNSUPPORTED', message, options);
  }
}

/** Unknown model ID, or a preview model used without `allowPreview`. */
export class ModelNotFoundError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_MODEL', message, options);
  }
}

/** A model was passed to a verb it does not serve. */
export class WrongVerbError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_VERB', message, options);
  }
}

/** Network, CORS or missing-file failure while downloading model files. */
export class DownloadError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_DOWNLOAD', message, { retryable: true, ...options });
  }
}

/** Not enough storage for the model files. */
export class StorageQuotaError extends EdgewiseError {
  readonly required: number | undefined;
  readonly available: number | undefined;
  constructor(message: string, options: EdgewiseErrorOptions & { required?: number; available?: number } = {}) {
    super('E_QUOTA', message, options);
    this.required = options.required;
    this.available = options.available;
  }
}

/** GPU allocation failed or the device was lost. */
export class OutOfMemoryError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_OOM', message, options);
  }
}

/** The inference runtime failed: unsupported operator, NaN output, and so on. */
export class BackendError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_BACKEND', message, options);
  }
}

/** Model output did not match the Zod schema passed to `generate`. */
export class SchemaValidationError extends EdgewiseError {
  readonly raw: string;
  readonly issues: unknown;
  constructor(message: string, options: EdgewiseErrorOptions & { raw: string; issues?: unknown }) {
    super('E_SCHEMA', message, options);
    this.raw = options.raw;
    this.issues = options.issues;
  }
}

/** Microphone or camera permission was denied. */
export class PermissionError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_PERMISSION', message, options);
  }
}

/** The caller's AbortSignal fired. */
export class AbortError extends EdgewiseError {
  constructor(message = 'The operation was aborted.', options?: EdgewiseErrorOptions) {
    super('E_ABORT', message, options);
  }
}

/** Invalid options passed to a verb or to `configure`. */
export class ConfigError extends EdgewiseError {
  constructor(message: string, options?: EdgewiseErrorOptions) {
    super('E_CONFIG', message, options);
  }
}

export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new AbortError(undefined, { cause: signal.reason });
  }
}

/** Turn an unknown runtime failure into a typed Edgewise error. */
export function toEdgewiseError(err: unknown, context: string): EdgewiseError {
  if (err instanceof EdgewiseError) return err;
  const message = err instanceof Error ? err.message : String(err);
  if (err instanceof Error && (err.name === 'AbortError' || /aborted/i.test(message))) {
    return new AbortError(message, { cause: err });
  }
  if (/out of memory|OOM|allocation failed|device (was )?lost|bad_alloc/i.test(message)) {
    return new OutOfMemoryError(`${context}: ${message}`, {
      cause: err,
      hint: 'The model did not fit in memory. Try a smaller model or dtype, or lower the GPU budget.',
    });
  }
  if (/quota|QuotaExceeded|ENOSPC|no space left/i.test(message)) {
    return new StorageQuotaError(`${context}: ${message}`, {
      cause: err,
      hint: 'There is not enough storage for this model. Delete cached models or call persist().',
    });
  }
  if (
    /fetch failed|failed to fetch|networkerror|network|load failed|ECONNRESET|ETIMEDOUT|ENOTFOUND|Could not locate file|status: 40|status: 50/i.test(message)
  ) {
    return new DownloadError(`${context}: ${message}`, {
      cause: err,
      hint: 'A model file could not be downloaded. Check the network connection and try again.',
    });
  }
  return new BackendError(`${context}: ${message}`, {
    cause: err,
    hint: 'The inference runtime failed. Try device: "wasm" or "cpu", or report the issue.',
  });
}
