import { capabilities, cloneVoice, generate, speak } from '../../src/index.ts';
import { resample } from '../../src/platform/audio.ts';
import { isBrowser, it2, on, report, suite } from './setup.ts';

// Chatterbox needs WebGPU in browsers (its quantized graphs use an operator the WebAssembly build lacks).
const device = isBrowser ? ('webgpu' as const) : undefined;
const consent = { attested: true as const, by: 'test suite (synthetic Kokoro voices)' };

/** Median pitch in Hz over voiced frames, by autocorrelation. */
function pitch(x: Float32Array, sr: number): number {
  const out: number[] = [];
  const N = 1024;
  for (let s = 0; s + N < x.length; s += 512) {
    const fr = x.subarray(s, s + N);
    let e = 0;
    for (const v of fr) e += v * v;
    if (e / N < 1e-4) continue;
    let best = 0;
    let lag = 0;
    for (let l = Math.floor(sr / 400); l < sr / 70; l++) {
      let c = 0;
      for (let i = 0; i + l < N; i++) c += fr[i] * fr[i + l];
      if (c > best) {
        best = c;
        lag = l;
      }
    }
    if (lag) out.push(sr / lag);
  }
  out.sort((a, b) => a - b);
  return out[Math.floor(out.length / 2)] ?? 0;
}

const reference = (voice: string) =>
  speak({ model: 'kokoro-82m', voice, input: 'This is a short sample of my voice, recorded for cloning. I speak clearly and at a normal pace.' });

suite('speak · voice cloning (Chatterbox Turbo)', () => {
  it2(on(['bun', 'node', 'browser'], 'clones two voices and keeps the words and the pitch'), async () => {
    // In browsers Chatterbox needs a hardware GPU: on a software WebGPU adapter it exceeds typical CI memory.
    if (isBrowser && !(await capabilities()).hardwareGpu) {
      report('chatterbox', 'skipped: no hardware GPU in this browser');
      return;
    }
    const pitches: Record<string, number> = {};
    for (const v of ['af_heart', 'bm_george']) {
      const ref = await reference(v);
      const out = await speak({
        model: 'voice:clone',
        allowPreview: true,
        device,
        input: 'Hello! This voice was cloned on the device.',
        voice: { reference: ref.samples, sampleRate: ref.sampleRate, consent },
      });
      report(`chatterbox ${v}`, out.info);
      expect(out.sampleRate).toBe(24000);
      expect(out.duration).toBeGreaterThan(1.5);
      const heard = await generate({ model: 'moonshine-base', input: resample(out.samples, 24000, 16000) });
      expect(heard.text.toLowerCase()).toMatch(/clon/);
      pitches[v] = pitch(out.samples, 24000);
    }
    report('clone pitch', pitches);
    expect(pitches.af_heart).toBeGreaterThan(pitches.bm_george * 1.2);
  });

  it2(on(['bun', 'node'], 'reuses a voice from cloneVoice() and from saveAs'), async () => {
    const ref = await reference('bm_george');
    const voice = await cloneVoice({ reference: ref.samples, sampleRate: ref.sampleRate, consent, saveAs: 'edgewise-test-george', allowPreview: true });
    expect(voice.kind).toBe('cloned-voice');
    const a = await speak({ model: 'voice:clone', allowPreview: true, input: 'First sentence. Second sentence.', voice });
    const b = await speak({ model: 'voice:clone', allowPreview: true, input: 'First sentence. Second sentence.', voice: 'saved:edgewise-test-george' });
    // Greedy decoding with the same conditioning gives the same speech tokens, so the same length.
    // The decoder adds sampling noise, so the waveforms differ slightly.
    expect(b.samples.length).toBe(a.samples.length);
    const [pa, pb] = [pitch(a.samples, 24000), pitch(b.samples, 24000)];
    expect(Math.abs(pa - pb) / pa).toBeLessThan(0.15);
  });
});
