import { FRAME, VadSegmenter } from '../../src/backends/vad.ts';

const frame = () => new Float32Array(FRAME);

function run(probs: number[], o = {}) {
  const seg = new VadSegmenter(o);
  const events: string[] = [];
  const lengths: number[] = [];
  for (const p of probs) {
    const e = seg.push(frame(), p);
    if (e) {
      events.push(e.type);
      if (e.type === 'end') lengths.push(e.audio.length / FRAME);
    }
  }
  const last = seg.flush();
  if (last) {
    events.push(last.type);
    if (last.type === 'end') lengths.push(last.audio.length / FRAME);
  }
  return { events, lengths };
}

describe('VAD segmentation', () => {
  it('finds one utterance with padding before speech', () => {
    const probs = [...Array(20).fill(0.05), ...Array(30).fill(0.9), ...Array(40).fill(0.05)];
    const r = run(probs, { redemptionMs: 320, preSpeechPadMs: 96 });
    expect(r.events).toEqual(['start', 'end']);
    // 3 pre-speech frames + 1 trigger + 29 speech + 10 redemption frames
    expect(r.lengths[0]).toBe(4 + 29 + 10);
  });

  it('bridges short pauses', () => {
    const probs = [...Array(20).fill(0.9), ...Array(5).fill(0.1), ...Array(20).fill(0.9), ...Array(40).fill(0.1)];
    expect(run(probs, { redemptionMs: 320 }).events).toEqual(['start', 'end']);
  });

  it('splits on long pauses', () => {
    const probs = [...Array(20).fill(0.9), ...Array(30).fill(0.1), ...Array(20).fill(0.9), ...Array(30).fill(0.1)];
    expect(run(probs, { redemptionMs: 320 }).events).toEqual(['start', 'end', 'start', 'end']);
  });

  it('drops blips shorter than the minimum', () => {
    const probs = [...Array(10).fill(0.1), 0.9, 0.9, ...Array(30).fill(0.1)];
    expect(run(probs, { minSpeechMs: 250, redemptionMs: 320 }).events).toEqual(['start', 'misfire']);
  });

  it('keeps frames between the two thresholds as speech', () => {
    const probs = [...Array(10).fill(0.9), ...Array(40).fill(0.4), ...Array(20).fill(0.1)];
    const r = run(probs, { redemptionMs: 320 });
    expect(r.events).toEqual(['start', 'end']);
    expect(r.lengths[0]).toBeGreaterThan(50);
  });

  it('closes an open utterance at the end of the stream', () => {
    expect(run(Array(20).fill(0.9)).events).toEqual(['start', 'end']);
  });
});
