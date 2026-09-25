import {
  AbortError,
  BackendError,
  DownloadError,
  EdgewiseError,
  OutOfMemoryError,
  StorageQuotaError,
  toEdgewiseError,
  UnsupportedInputError,
} from '../../src/core/errors.ts';

describe('errors', () => {
  it('carry a stable code, hint and retryable flag', () => {
    const e = new UnsupportedInputError('no images here', { hint: 'use a vision model' });
    expect(e).toBeInstanceOf(EdgewiseError);
    expect(e).toBeInstanceOf(Error);
    expect(e.code).toBe('E_INPUT');
    expect(e.hint).toBe('use a vision model');
    expect(e.retryable).toBe(false);
    expect(e.name).toBe('UnsupportedInputError');
  });

  it('defaults the hint to the message', () => {
    expect(new BackendError('boom').hint).toBe('boom');
  });

  it('marks download errors retryable by default', () => {
    expect(new DownloadError('x').retryable).toBe(true);
    expect(new DownloadError('x', { retryable: false }).retryable).toBe(false);
  });

  it('keeps the cause', () => {
    const cause = new Error('inner');
    expect(new BackendError('outer', { cause }).cause).toBe(cause);
  });

  it('maps runtime failures to typed errors', () => {
    expect(toEdgewiseError(new Error('failed to allocate: out of memory'), 'x')).toBeInstanceOf(OutOfMemoryError);
    expect(toEdgewiseError(new Error('QuotaExceededError'), 'x')).toBeInstanceOf(StorageQuotaError);
    expect(toEdgewiseError(new Error('fetch failed'), 'x')).toBeInstanceOf(DownloadError);
    const abort = new Error('The operation was aborted');
    abort.name = 'AbortError';
    expect(toEdgewiseError(abort, 'x')).toBeInstanceOf(AbortError);
    expect(toEdgewiseError('weird', 'ctx')).toBeInstanceOf(BackendError);
    expect(toEdgewiseError('weird', 'ctx').message).toContain('ctx');
  });

  it('passes Edgewise errors through untouched', () => {
    const e = new UnsupportedInputError('x');
    expect(toEdgewiseError(e, 'y')).toBe(e);
  });
});
