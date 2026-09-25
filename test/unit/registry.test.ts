import { configure, resetConfig } from '../../src/core/config.ts';
import { ConfigError, ModelNotFoundError, UnsupportedInputError, WrongVerbError } from '../../src/core/errors.ts';
import { defineModel, registry, resolveManifest } from '../../src/core/registry.ts';

afterEach(() => resetConfig());

describe('registry', () => {
  it('resolves aliases to manifests', () => {
    expect(registry.get('text:default').id).toBe('lfm2.5-350m');
    expect(registry.get('embed:tiny').verb).toBe('embed');
    expect(registry.aliases()['forecast:default']).toBe('chronos-bolt-tiny');
  });

  it('throws a helpful error for unknown models', () => {
    expect(() => registry.get('lfm2.5-350')).toThrow(ModelNotFoundError);
    try {
      registry.get('lfm2.5-350');
    } catch (e) {
      expect((e as Error).message).toContain('Did you mean');
    }
  });

  it('pins every Hugging Face model to a full commit SHA', () => {
    for (const m of registry.list()) {
      if ('repo' in m.source) expect(m.source.revision).toMatch(/^[0-9a-f]{40}$/);
      if ('bucket' in m.source) expect(m.config?.sha256).toBeTruthy();
    }
  });

  it('checks bucket-hosted files with a SHA-256', () => {
    for (const m of registry.list()) {
      if (!('bucket' in m.source)) continue;
      const files = Object.values((m.config?.files ?? { f: m.config?.file }) as Record<string, string>);
      const sums = m.config?.sha256 as Record<string, string>;
      for (const f of files) expect(sums[f]).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it('gives every model a licence, a status and at least one variant', () => {
    for (const m of registry.list()) {
      expect(m.license).toBeTruthy();
      expect(['stable', 'preview', 'experimental']).toContain(m.status);
      expect(m.variants.length).toBeGreaterThan(0);
      expect(m.accepts.length).toBeGreaterThan(0);
    }
  });

  it('filters by verb and accepted input', () => {
    const audio = registry.list({ verb: 'generate', accepts: 'audio' });
    expect(audio.length).toBeGreaterThan(0);
    expect(audio.every((m) => m.accepts.includes('audio'))).toBe(true);
    expect(registry.list({ verb: 'paint' }).map((m) => m.id)).toContain('sd-turbo');
  });

  it('refuses a model on the wrong verb', () => {
    expect(() => resolveManifest('kokoro-82m', { verb: 'generate' })).toThrow(WrongVerbError);
  });

  it('refuses inputs the model does not accept, and suggests one that does', () => {
    try {
      resolveManifest('lfm2.5-350m', { verb: 'generate', inputs: ['image'] });
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(UnsupportedInputError);
      expect((e as UnsupportedInputError).hint).toMatch(/Try "/);
    }
  });

  it('gates preview models behind allowPreview', () => {
    expect(() => resolveManifest('qwen3-0.6b', { verb: 'generate' })).toThrow(ModelNotFoundError);
    expect(resolveManifest('qwen3-0.6b', { verb: 'generate', allowPreview: true }).id).toBe('qwen3-0.6b');
    configure({ allowPreview: true });
    expect(resolveManifest('qwen3-0.6b', { verb: 'generate' }).id).toBe('qwen3-0.6b');
  });

  it('enforces the licence allow-list', () => {
    configure({ licenses: ['apache-2.0'] });
    expect(() => resolveManifest('lfm2.5-350m', { verb: 'generate' })).toThrow(/licence/);
    expect(resolveManifest('all-minilm-l6-v2', { verb: 'embed' }).id).toBe('all-minilm-l6-v2');
  });

  it('registers custom models and aliases', () => {
    const m = defineModel({ id: 'acme/test-embed', verb: 'embed', accepts: ['text'], task: 'feature-extraction', source: { url: 'https://example.com/m' } });
    expect(registry.get('acme/test-embed')).toBe(m);
    registry.alias('embed:acme', 'acme/test-embed');
    expect(registry.get('embed:acme').id).toBe('acme/test-embed');
    expect(() => defineModel({ id: 'x', verb: 'embed', accepts: [], task: 'feature-extraction', source: { url: 'u' } })).toThrow(ConfigError);
  });
});
