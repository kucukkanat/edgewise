import { UnsupportedInputError } from '../core/errors.ts';
import type { DecodedAudio } from './types.ts';

/** Parse a RIFF/WAVE file into mono float samples. Supports PCM 8/16/24/32-bit and float32/64. */
export function decodeWav(bytes: Uint8Array): DecodedAudio {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
  if (bytes.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
    throw new UnsupportedInputError('Not a WAV file.', {
      hint: 'On Bun and Node, pass WAV audio or a Float32Array of samples. Other formats need decoding first.',
    });
  }
  let offset = 12;
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= bytes.byteLength) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === 'fmt ') {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
      if (format === 0xfffe && size >= 26) format = view.getUint16(body + 24, true);
    } else if (id === 'data') {
      dataOffset = body;
      dataSize = Math.min(size, bytes.byteLength - body);
      break;
    }
    offset = body + size + (size % 2);
  }
  if (dataOffset < 0 || !channels || !sampleRate) throw new UnsupportedInputError('WAV file has no audio data.');
  const bytesPer = bits / 8;
  const frames = Math.floor(dataSize / (bytesPer * channels));
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const p = dataOffset + (i * channels + c) * bytesPer;
      let v: number;
      if (format === 3) v = bits === 64 ? view.getFloat64(p, true) : view.getFloat32(p, true);
      else if (bits === 8) v = (view.getUint8(p) - 128) / 128;
      else if (bits === 16) v = view.getInt16(p, true) / 32768;
      else if (bits === 24) {
        const x = view.getUint8(p) | (view.getUint8(p + 1) << 8) | (view.getInt8(p + 2) << 16);
        v = x / 8388608;
      } else if (bits === 32) v = view.getInt32(p, true) / 2147483648;
      else throw new UnsupportedInputError(`Unsupported WAV bit depth: ${bits}.`);
      sum += v;
    }
    out[i] = sum / channels;
  }
  return { samples: out, sampleRate };
}

/** Encode mono float samples as a 16-bit PCM WAV file. */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const dataBytes = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, 'RIFF');
  v.setUint32(4, 36 + dataBytes, true);
  w(8, 'WAVE');
  w(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, 'data');
  v.setUint32(40, dataBytes, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buf);
}

/** Resample mono audio with linear interpolation (anti-aliased by a box filter when downsampling). */
export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples;
  const ratio = from / to;
  const out = new Float32Array(Math.max(1, Math.floor(samples.length / ratio)));
  if (ratio > 1) {
    for (let i = 0; i < out.length; i++) {
      const start = i * ratio;
      const end = Math.min(samples.length, start + ratio);
      let sum = 0;
      let n = 0;
      for (let j = Math.floor(start); j < end; j++) {
        sum += samples[j];
        n++;
      }
      out[i] = n ? sum / n : 0;
    }
  } else {
    for (let i = 0; i < out.length; i++) {
      const x = i * ratio;
      const i0 = Math.floor(x);
      const f = x - i0;
      out[i] = (samples[i0] ?? 0) * (1 - f) + (samples[i0 + 1] ?? samples[i0] ?? 0) * f;
    }
  }
  return out;
}

export function isWav(bytes: Uint8Array): boolean {
  return bytes.length > 12 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46;
}
