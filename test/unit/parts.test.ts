import { ConfigError, UnsupportedInputError } from '../../src/core/errors.ts';
import { messageText, normalizeInput, toPart } from '../../src/core/parts.ts';

const pixels = { data: new Uint8ClampedArray(4 * 4 * 3), width: 4, height: 4, channels: 3 as const };

describe('input parts', () => {
  it('detects part types from plain values', () => {
    expect(toPart('hi')).toEqual({ type: 'text', text: 'hi' });
    expect(toPart(new Float32Array(10)).type).toBe('audio');
    expect(toPart(pixels).type).toBe('image');
    expect(toPart(new Blob([], { type: 'image/png' })).type).toBe('image');
    expect(toPart(new Blob([], { type: 'audio/wav' })).type).toBe('audio');
    expect(toPart(new Blob([], { type: 'video/mp4' })).type).toBe('video');
    expect(toPart(new URL('https://x.test/a.jpg')).type).toBe('image');
    expect(toPart(new URL('https://x.test/a.wav')).type).toBe('audio');
  });

  it('keeps explicit parts as they are', () => {
    const p = { type: 'audio' as const, audio: new Float32Array(4), sampleRate: 44100 };
    expect(toPart(p)).toBe(p);
  });

  it('refuses things it cannot classify, with a hint', () => {
    expect(() => toPart(new Blob([]))).toThrow(UnsupportedInputError);
    expect(() => toPart(new URL('https://x.test/unknown'))).toThrow(UnsupportedInputError);
    expect(() => toPart(42 as never)).toThrow(UnsupportedInputError);
  });

  it('normalizes input into one user message and lists the types', () => {
    const n = normalizeInput({ input: [pixels, 'What is this?'], system: 'Be brief.' });
    expect(n.messages).toHaveLength(2);
    expect(n.messages[0]).toEqual({ role: 'system', content: 'Be brief.' });
    expect(n.types.sort()).toEqual(['image', 'text']);
  });

  it('keeps a single string as plain content', () => {
    expect(normalizeInput({ input: 'hi' }).messages[0].content).toBe('hi');
  });

  it('rejects input and messages together, or neither', () => {
    expect(() => normalizeInput({ input: 'a', messages: [] })).toThrow(ConfigError);
    expect(() => normalizeInput({})).toThrow(ConfigError);
    expect(() => normalizeInput({ input: [] })).toThrow(ConfigError);
  });

  it('recognizes live audio sources', () => {
    const src = { kind: 'audio-source' as const, sampleRate: 16000, utterances: async function* () {} };
    const n = normalizeInput({ input: src });
    expect(n.source).toBe(src);
    expect(n.types).toContain('audio');
  });

  it('reads the text of mixed messages', () => {
    expect(
      messageText({
        role: 'user',
        content: [
          { type: 'text', text: 'a' },
          { type: 'image', image: pixels },
          { type: 'text', text: 'b' },
        ],
      }),
    ).toBe('a\nb');
  });
});
