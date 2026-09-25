import { UnsupportedInputError } from '../../src/core/errors.ts';
import { decodeWav, encodeWav, isWav, resample } from '../../src/platform/audio.ts';

describe('audio utilities', () => {
  it('round-trips 16-bit WAV', () => {
    const s = Float32Array.from({ length: 1600 }, (_, i) => Math.sin(i / 10) * 0.5);
    const wav = encodeWav(s, 16000);
    expect(isWav(wav)).toBe(true);
    const d = decodeWav(wav);
    expect(d.sampleRate).toBe(16000);
    expect(d.samples.length).toBe(1600);
    let err = 0;
    for (let i = 0; i < s.length; i++) err = Math.max(err, Math.abs(d.samples[i] - s[i]));
    expect(err).toBeLessThan(1e-4);
  });

  it('clamps out-of-range samples', () => {
    const d = decodeWav(encodeWav(Float32Array.from([2, -2]), 8000));
    expect(d.samples[0]).toBeCloseTo(1, 3);
    expect(d.samples[1]).toBeCloseTo(-1, 3);
  });

  it('mixes stereo float WAV down to mono', () => {
    const frames = 4;
    const buf = new ArrayBuffer(44 + frames * 2 * 4);
    const v = new DataView(buf);
    const w = (o: number, t: string) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    w(0, 'RIFF');
    v.setUint32(4, 36 + frames * 8, true);
    w(8, 'WAVE');
    w(12, 'fmt ');
    v.setUint32(16, 16, true);
    v.setUint16(20, 3, true);
    v.setUint16(22, 2, true);
    v.setUint32(24, 22050, true);
    v.setUint32(28, 22050 * 8, true);
    v.setUint16(32, 8, true);
    v.setUint16(34, 32, true);
    w(36, 'data');
    v.setUint32(40, frames * 8, true);
    for (let i = 0; i < frames; i++) {
      v.setFloat32(44 + i * 8, 1, true);
      v.setFloat32(48 + i * 8, 0, true);
    }
    const d = decodeWav(new Uint8Array(buf));
    expect(d.sampleRate).toBe(22050);
    expect(Array.from(d.samples)).toEqual([0.5, 0.5, 0.5, 0.5]);
  });

  it('refuses non-WAV data', () => {
    expect(() => decodeWav(new Uint8Array([1, 2, 3]))).toThrow(UnsupportedInputError);
  });

  it('resamples to the right length', () => {
    expect(resample(new Float32Array(48000), 48000, 16000).length).toBe(16000);
    expect(resample(new Float32Array(16000), 16000, 24000).length).toBe(24000);
    const same = new Float32Array(10);
    expect(resample(same, 16000, 16000)).toBe(same);
  });

  it('keeps a DC signal level when resampling', () => {
    const r = resample(new Float32Array(4410).fill(0.25), 44100, 16000);
    expect(Math.max(...r)).toBeCloseTo(0.25, 5);
    expect(Math.min(...r)).toBeCloseTo(0.25, 5);
  });
});
