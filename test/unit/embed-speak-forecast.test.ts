import { prepareContext } from '../../src/backends/chronos.ts';
import { ConfigError } from '../../src/core/errors.ts';
import { decodeWav } from '../../src/platform/audio.ts';
import { mockModel } from '../../src/test/index.ts';
import { embed } from '../../src/verbs/embed.ts';
import { forecast, resamplePoints } from '../../src/verbs/forecast.ts';
import { cloneVoice, speak } from '../../src/verbs/speak.ts';

describe('embed (mock)', () => {
  const model = mockModel({ verb: 'embed', respond: ({ texts }) => texts.map((t) => [t.length, 1, 0]) });
  it('embeds one value or many', async () => {
    const one = await embed({ model, input: 'abc' });
    expect(Array.from(one.embedding)).toEqual([3, 1, 0]);
    expect(one.dimensions).toBe(3);
    const many = await embed({ model, values: ['a', 'bb'] });
    expect(many.embeddings).toHaveLength(2);
  });
  it('needs exactly one of input and values', async () => {
    await expect(embed({ model, input: 'a', values: ['b'] } as never)).rejects.toBeInstanceOf(ConfigError);
  });
  it('only shortens vectors on models trained for it', async () => {
    await expect(embed({ model: 'all-minilm-l6-v2', input: 'x', dimensions: 128 })).rejects.toThrow(/cannot shorten/);
    await expect(embed({ model: 'embeddinggemma-300m', input: 'x', dimensions: 100 })).rejects.toThrow(/supports dimensions/);
  });
});

describe('speak (mock)', () => {
  const model = mockModel({ verb: 'speak' });
  it('synthesizes sentence by sentence and returns WAV-ready audio', async () => {
    const run = speak({ model, input: 'One. Two three.' });
    const chunks: string[] = [];
    for await (const c of run) chunks.push(c.text);
    const audio = await run;
    expect(chunks).toEqual(['One.', 'Two three.']);
    expect(audio.samples.length).toBeGreaterThan(0);
    const wav = decodeWav(audio.toWav());
    expect(wav.samples.length).toBe(audio.samples.length);
  });
  it('accepts a stream of text', async () => {
    async function* deltas() {
      yield 'Hel';
      yield 'lo there. And ';
      yield 'bye.';
    }
    const chunks: string[] = [];
    for await (const c of speak({ model, input: deltas() })) chunks.push(c.text);
    expect(chunks).toEqual(['Hello there.', 'And bye.']);
  });
  it('checks speed and cloning requests', async () => {
    await expect(Promise.resolve(speak({ model, input: 'x', speed: 5 }))).rejects.toBeInstanceOf(ConfigError);
    await expect(
      Promise.resolve(speak({ model: 'kokoro-82m', input: 'x', voice: { reference: new Float32Array(1), consent: { attested: true } } })),
    ).rejects.toThrow(/clone/);
  });
});

describe('forecast (mock)', () => {
  const model = mockModel({ verb: 'forecast', respond: ({ series, horizon }) => Array(horizon).fill(series[series.length - 1]) });
  it('returns median, mean and requested quantiles', async () => {
    const r = await forecast({ model, series: [1, 2, 3], horizon: 4, quantiles: [0.1, 0.5, 0.9] });
    expect(Array.from(r.median)).toEqual([3, 3, 3, 3]);
    expect(Object.keys(r.quantiles).map(Number)).toEqual([0.1, 0.5, 0.9]);
  });
  it('forecasts several series at once', async () => {
    const r = await forecast({
      model,
      series: [
        [1, 2],
        [5, 6, 7],
      ],
      horizon: 2,
    });
    expect(r.forecasts.map((f) => f.median[0])).toEqual([2, 7]);
  });
  it('resamples timestamped points and returns forecast times', async () => {
    const t0 = Date.UTC(2026, 0, 1);
    const pts = [0, 1, 2, 4].map((h) => ({ t: t0 + h * 3600e3, v: h }));
    const r = await forecast({ model, series: pts, horizon: 2 });
    expect(r.times).toEqual([t0 + 5 * 3600e3, t0 + 6 * 3600e3]);
  });
  it('rejects quantiles the model cannot produce', async () => {
    await expect(forecast({ model: 'chronos-bolt-tiny', series: [1, 2], horizon: 2, quantiles: [0.01] })).rejects.toThrow(/outside/);
    await expect(forecast({ model, series: [1, 2], horizon: 0 })).rejects.toBeInstanceOf(ConfigError);
  });
});

describe('Chronos preprocessing', () => {
  it('normalizes, pads to whole patches and masks missing values', () => {
    const series = Float32Array.from([1, 2, 3, Number.NaN, 5]);
    const p = prepareContext(series, { contextLength: 2048, patchSize: 4 });
    expect(p.patches).toBe(2);
    expect(p.loc).toBeCloseTo(2.75);
    // first patch: 3 padded values then the first observation
    expect(Array.from(p.pc.subarray(4, 8))).toEqual([0, 0, 0, 1]);
    expect(p.am[0]).toBe(1);
    // the NaN is masked out in the second patch
    expect(p.pc[8 + 4 + 2]).toBe(0);
  });
  it('keeps only the most recent context', () => {
    const p = prepareContext(new Float32Array(100).fill(1), { contextLength: 32, patchSize: 16 });
    expect(p.patches).toBe(2);
    expect(p.scale).toBe(1e-5);
  });
  it('resamples uneven timestamps onto a grid', () => {
    const r = resamplePoints([
      { t: 0, v: 0 },
      { t: 10, v: 1 },
      { t: 30, v: 3 },
    ]);
    expect(r.step).toBe(10);
    expect(Array.from(r.values)).toEqual([0, 1, 2, 3]);
  });
});

describe('PNG encoder', () => {
  it('writes a valid PNG signature, header and chunks', async () => {
    const { encodePng } = await import('../../src/core/png.ts');
    const px = new Uint8Array(2 * 2 * 4).fill(200);
    const png = await encodePng(px, 2, 2);
    expect(Array.from(png.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    expect(new TextDecoder().decode(png.subarray(12, 16))).toBe('IHDR');
    const v = new DataView(png.buffer, png.byteOffset);
    expect(v.getUint32(16)).toBe(2);
    expect(new TextDecoder().decode(png.subarray(png.length - 8, png.length - 4))).toBe('IEND');
  });
});

describe('voice cloning (no downloads)', () => {
  it('needs consent and a reference', async () => {
    const ref = new Float32Array(24000 * 4);
    await expect(
      Promise.resolve(speak({ model: 'chatterbox-turbo', allowPreview: true, input: 'Hi.', voice: { reference: ref, consent: {} as never } })),
    ).rejects.toThrow(/consent/);
    await expect(Promise.resolve(speak({ model: 'chatterbox-turbo', allowPreview: true, input: 'Hi.' }))).rejects.toThrow(/reference audio/);
    await expect(Promise.resolve(speak({ model: 'chatterbox-turbo', allowPreview: true, input: 'Hi.', voice: 'saved:nobody-here' }))).rejects.toThrow(
      /No saved voice/,
    );
    await expect(cloneVoice({ model: 'kokoro-82m', reference: ref, consent: { attested: true } })).rejects.toThrow(/does not accept audio/);
    await expect(cloneVoice({ reference: ref, consent: { attested: false } as never })).rejects.toThrow(/consent/);
  });

  it('saves and restores a cloned voice byte for byte', async () => {
    const { voiceFromBytes, voiceToBytes } = await import('../../src/backends/chatterbox.ts');
    const v = {
      kind: 'cloned-voice' as const,
      model: 'chatterbox-turbo',
      audio_features: { type: 'float32' as const, dims: [1, 2, 3], data: Float32Array.from([1, 2, 3, 4, 5, 6]) },
      audio_tokens: { type: 'int64' as const, dims: [1, 3], data: BigInt64Array.from([7n, 8n, 9n]) },
      speaker_embeddings: { type: 'float32' as const, dims: [1, 2], data: Float32Array.from([0.5, -0.5]) },
      speaker_features: { type: 'float32' as const, dims: [1, 1, 1], data: Float32Array.from([42]) },
    };
    const back = voiceFromBytes(voiceToBytes(v));
    expect(back.model).toBe('chatterbox-turbo');
    expect(Array.from(back.audio_tokens.data as BigInt64Array)).toEqual([7n, 8n, 9n]);
    expect(back.audio_features.dims).toEqual([1, 2, 3]);
    expect(Array.from(back.speaker_features.data as Float32Array)).toEqual([42]);
  });
});

describe('saved voice validation', () => {
  it('rejects damaged files with ConfigError', async () => {
    const { voiceFromBytes, voiceToBytes } = await import('../../src/backends/chatterbox.ts');
    const t = (n: number) => ({ type: 'float32' as const, dims: [n], data: new Float32Array(n) });
    const good = voiceToBytes({
      kind: 'cloned-voice',
      model: 'm',
      audio_features: t(2),
      audio_tokens: { type: 'int64', dims: [1], data: new BigInt64Array(1) },
      speaker_embeddings: t(1),
      speaker_features: t(1),
    });
    expect(() => voiceFromBytes(good.subarray(0, good.length - 3))).toThrow(ConfigError);
    expect(() => voiceFromBytes(new Uint8Array(5))).toThrow(ConfigError);
    const corrupt = good.slice();
    corrupt[14] = 0x7b;
    corrupt[15] = 0x7b;
    expect(() => voiceFromBytes(corrupt)).toThrow(ConfigError);
  });
});
