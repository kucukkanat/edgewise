import { detectSpeech } from '../../src/helpers/index.ts';
import { audioSource, generate, speak } from '../../src/index.ts';
import { resample } from '../../src/platform/audio.ts';
import { it2, on, report, suite } from './setup.ts';

const all = ['bun', 'node', 'browser'];
let spoken: { samples: Float32Array; sampleRate: number; wav: Uint8Array } | undefined;

async function hello() {
  if (!spoken) {
    const a = await speak({ model: 'kokoro-82m', voice: 'af_heart', input: 'Hello from Edgewise. This sentence was spoken on the device.' });
    report('kokoro', a.info);
    spoken = { samples: a.samples, sampleRate: a.sampleRate, wav: a.toWav() };
  }
  return spoken;
}

suite('speak, STT and VAD · real models', () => {
  it2(on(all, 'speaks text with Kokoro'), async () => {
    const a = await hello();
    expect(a.sampleRate).toBe(24000);
    const seconds = a.samples.length / a.sampleRate;
    expect(seconds).toBeGreaterThan(2);
    expect(seconds).toBeLessThan(10);
    let peak = 0;
    for (const v of a.samples) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeGreaterThan(0.05);
  });

  it2(on(all, 'speaks a stream of text sentence by sentence'), async () => {
    const deltas = (async function* () {
      yield 'First part. ';
      yield 'Second ';
      yield 'part here.';
    })();
    const chunks: string[] = [];
    for await (const c of speak({ model: 'kokoro-82m', input: deltas })) chunks.push(c.text);
    expect(chunks).toEqual(['First part.', 'Second part here.']);
  });

  it2(on(all, 'transcribes speech with Moonshine (round trip)'), async () => {
    const a = await hello();
    const r = await generate({ model: 'moonshine-tiny', input: new Blob([a.wav as Uint8Array<ArrayBuffer>], { type: 'audio/wav' }) });
    report('moonshine-tiny', r.info);
    expect(r.text.toLowerCase()).toMatch(/hello/);
    expect(r.text.toLowerCase()).toMatch(/device/);
  });

  it2(on(all, 'transcribes speech with Moonshine Base'), async () => {
    const a = await hello();
    const r = await generate({ model: 'moonshine-base', input: { type: 'audio', audio: a.samples, sampleRate: a.sampleRate } });
    report('moonshine-base', r.info);
    expect(r.text.toLowerCase()).toMatch(/hello/);
  });

  it2(on(all, 'transcribes long audio window by window with progress and timestamps'), async () => {
    const a = await hello();
    const speech = resample(a.samples, a.sampleRate, 16000);
    // 75 s: the sentence every 8 s, so Moonshine would be far past its ~30 s limit in one pass.
    const clip = new Float32Array(16000 * 75);
    for (let t = 1; t + 8 <= 75; t += 8) clip.set(speech.subarray(0, Math.min(speech.length, 16000 * 6)), t * 16000);
    const progress: number[] = [];
    const r = await generate({
      model: 'moonshine-tiny',
      input: { type: 'audio', audio: clip, sampleRate: 16000 },
      timestamps: 'segment',
      onSegment: (_s, p) => progress.push(p.done / p.total),
    });
    report('moonshine-tiny long', r.info);
    expect(progress.length).toBeGreaterThan(2);
    expect(progress.at(-1)).toBe(1);
    expect((r.text.toLowerCase().match(/hello/g) ?? []).length).toBeGreaterThanOrEqual(6);
    const starts = r.segments?.map((s) => s.start) ?? [];
    expect(starts.length).toBeGreaterThan(2);
    for (let i = 1; i < starts.length; i++) expect(starts[i]).toBeGreaterThan(starts[i - 1]);
    expect(r.segments?.at(-1)?.end).toBeLessThanOrEqual(75);
  });

  it2(on(all, 'transcribes with Whisper base and timestamps'), async () => {
    const a = await hello();
    const r = await generate({
      model: 'whisper-base',
      input: { type: 'audio', audio: a.samples, sampleRate: a.sampleRate },
      timestamps: 'segment',
      language: 'en',
    });
    report('whisper-base', r.info);
    expect(r.text.toLowerCase()).toMatch(/hello/);
    expect(r.segments?.length).toBeGreaterThan(0);
  });

  it2(on(all, 'finds speech with Silero VAD'), async () => {
    const a = await hello();
    const speech = resample(a.samples, a.sampleRate, 16000);
    const clip = new Float32Array(16000 * 5);
    clip.set(speech.subarray(0, Math.min(speech.length, 16000 * 3)), 16000);
    const segs = await detectSpeech(clip, { sampleRate: 16000 });
    expect(segs.length).toBeGreaterThan(0);
    expect(segs[0].start).toBeGreaterThan(0.7);
    expect(segs[0].start).toBeLessThan(1.4);
  });

  it2(on(all, 'reports speech start before the utterance ends (for barge-in)'), async () => {
    const a = await hello();
    const speech = resample(a.samples, a.sampleRate, 16000);
    const clip = new Float32Array(16000 * 5);
    clip.set(speech.subarray(0, Math.min(speech.length, 16000 * 3)), 16000);
    // Stream it in 100 ms chunks, like a microphone.
    async function* chunks() {
      for (let i = 0; i < clip.length; i += 1600) yield clip.subarray(i, i + 1600);
    }
    const events: string[] = [];
    let probs = 0;
    const src = audioSource(chunks(), {
      vad: { onSpeechStart: () => events.push('start'), onSpeechEnd: () => events.push('end'), onFrame: () => probs++ },
    });
    for await (const _ of src.utterances()) events.push('yield');
    expect(events.slice(0, 3)).toEqual(['start', 'end', 'yield']);
    expect(probs).toBeGreaterThan(100);
  });
});
