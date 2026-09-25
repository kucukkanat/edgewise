import { normalizeText, splitSentences } from '../../src/backends/kokoro-text.ts';

describe('speech text preparation', () => {
  it('reads numbers, money and times aloud', () => {
    expect(normalizeText('It costs $5.50')).toContain('5 dollars and 50 cents');
    expect(normalizeText('Meet at 3:05')).toContain('3 oh 5');
    expect(normalizeText('In 1984 we met')).toContain('19 84');
    expect(normalizeText('Pi is 3.14')).toContain('3 point 1 4');
    expect(normalizeText('Dr. Smith and Mr. Jones')).toBe('Doctor Smith and Mister Jones');
    expect(normalizeText('pages 10-12')).toBe('pages 10 to 12');
  });

  it('splits sentences and keeps the unfinished tail for streaming', () => {
    const r = splitSentences('Hello there. How are you? I am fi');
    expect(r.sentences).toEqual(['Hello there.', 'How are you?']);
    expect(r.rest).toBe('I am fi');
    expect(splitSentences('I am fine', true)).toEqual({ sentences: ['I am fine'], rest: '' });
  });

  it('does not split on abbreviations or decimals', () => {
    expect(splitSentences('Ask Dr. Who about it. Then go.', true).sentences).toEqual(['Ask Dr. Who about it.', 'Then go.']);
    expect(splitSentences('Version 2. is out. Yes!', true).sentences.length).toBeGreaterThanOrEqual(2);
  });

  it('waits for whitespace before closing a sentence mid-stream', () => {
    expect(splitSentences('Hello.').sentences).toEqual([]);
    expect(splitSentences('Hello. ').sentences).toEqual(['Hello.']);
  });
});
