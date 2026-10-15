import { splitOnSilence } from '../../src/backends/stt.ts';

const RATE = 16000;

/** A 220 Hz tone with silent gaps at the given [start, end] seconds. */
function speech(seconds: number, gaps: [number, number][] = []) {
  const a = new Float32Array(seconds * RATE);
  for (let i = 0; i < a.length; i++) a[i] = 0.3 * Math.sin((2 * Math.PI * 220 * i) / RATE);
  for (const [s, e] of gaps) a.fill(0, s * RATE, e * RATE);
  return a;
}

describe('splitOnSilence', () => {
  it('keeps short audio as one window', () => {
    expect(splitOnSilence(speech(20))).toEqual([{ start: 0, end: 20 * RATE }]);
  });

  it('cuts inside pauses, never past the longest window', () => {
    const w = splitOnSilence(
      speech(70, [
        [20, 21],
        [45, 46],
      ]),
    );
    expect(w).toHaveLength(3);
    expect(w[0].end / RATE).toBeGreaterThanOrEqual(20);
    expect(w[0].end / RATE).toBeLessThanOrEqual(21);
    expect(w[1].end / RATE).toBeGreaterThanOrEqual(45);
    expect(w[1].end / RATE).toBeLessThanOrEqual(46);
    for (const x of w) expect(x.end - x.start).toBeLessThanOrEqual(28 * RATE);
  });

  it('covers continuous speech with contiguous windows', () => {
    const w = splitOnSilence(speech(100));
    expect(w[0].start).toBe(0);
    expect(w.at(-1)?.end).toBe(100 * RATE);
    for (let i = 1; i < w.length; i++) expect(w[i].start).toBe(w[i - 1].end);
    for (const x of w) expect(x.end - x.start).toBeLessThanOrEqual(28 * RATE);
  });

  it('drops silent windows', () => {
    const w = splitOnSilence(speech(90, [[25, 70]]));
    for (const x of w) expect(x.start >= 30 * RATE && x.end <= 65 * RATE).toBe(false);
    expect(w.at(-1)?.end).toBe(90 * RATE);
  });

  it('trims silent edges around speech, keeping padding', () => {
    const a = new Float32Array(40 * RATE);
    a.set(speech(5), 10 * RATE);
    a.set(speech(5), 31 * RATE);
    const w = splitOnSilence(a);
    expect(w[0].start / RATE).toBeCloseTo(9.7, 1);
    expect(w.at(-1)?.end ?? 0).toBeLessThanOrEqual(36.3 * RATE);
    expect(w.at(-1)?.end ?? 0).toBeGreaterThanOrEqual(36 * RATE);
  });

  it('honours custom window lengths', () => {
    const w = splitOnSilence(speech(60), { maxSeconds: 10, minSeconds: 5 });
    for (const x of w) expect(x.end - x.start).toBeLessThanOrEqual(10 * RATE);
    expect(w.length).toBeGreaterThanOrEqual(6);
  });
});
