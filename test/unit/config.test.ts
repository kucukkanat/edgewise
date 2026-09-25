import { configure, getConfig, resetConfig } from '../../src/core/config.ts';
import { ConfigError } from '../../src/core/errors.ts';

afterEach(() => resetConfig());

describe('configure', () => {
  it('merges nested settings', () => {
    configure({ fallback: { onBackendError: 'throw' } });
    expect(getConfig().fallback.onBackendError).toBe('throw');
    expect(getConfig().fallback.onUnsupported).toBe('next-variant');
  });
  it('validates values', () => {
    expect(() => configure({ maxLoadedModels: 0 })).toThrow(ConfigError);
    expect(() => configure({ hub: 'ftp://x' })).toThrow(ConfigError);
  });
  it('resets to defaults', () => {
    configure({ hub: 'https://mirror.example.com' });
    resetConfig();
    expect(getConfig().hub).toBe('https://huggingface.co');
  });
});
