import { countWords } from '../../docs/words.ts';

describe('countWords', () => {
  it('spells counts for headlines', () => {
    expect(countWords(1)).toBe('One');
    expect(countWords(13)).toBe('Thirteen');
    expect(countWords(30)).toBe('Thirty');
    expect(countWords(32)).toBe('Thirty-two');
    expect(countWords(99)).toBe('Ninety-nine');
  });

  it('refuses counts it cannot spell', () => {
    for (const n of [0, 100, 2.5]) expect(() => countWords(n)).toThrow(/1 to 99/);
  });
});
