/**
 * Text normalization and phonemization for Kokoro.
 * Adapted from kokoro-js (Apache-2.0, https://github.com/hexgrad/kokoro), rewritten for readability.
 */
import { ConfigError } from '../core/errors.ts';

function splitNum(m: string): string {
  if (m.includes('.')) return m;
  if (m.includes(':')) {
    const [h, min] = m.split(':').map(Number);
    if (min === 0) return `${h} o'clock`;
    if (min < 10) return `${h} oh ${min}`;
    return `${h} ${min}`;
  }
  const year = Number.parseInt(m.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) return m;
  const left = m.slice(0, 2);
  const right = Number.parseInt(m.slice(2, 4), 10);
  const s = m.endsWith('s') ? 's' : '';
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) return `${left} hundred${s}`;
    if (right < 10) return `${left} oh ${right}${s}`;
  }
  return `${left} ${right}${s}`;
}

function money(m: string): string {
  const unit = m[0] === '$' ? 'dollar' : 'pound';
  if (Number.isNaN(Number(m.slice(1)))) return `${m.slice(1)} ${unit}s`;
  if (!m.includes('.')) return `${m.slice(1)} ${unit}${m.slice(1) === '1' ? '' : 's'}`;
  const [whole, frac] = m.slice(1).split('.');
  const cents = Number.parseInt(frac.padEnd(2, '0'), 10);
  const sub = m[0] === '$' ? (cents === 1 ? 'cent' : 'cents') : cents === 1 ? 'penny' : 'pence';
  return `${whole} ${unit}${whole === '1' ? '' : 's'} and ${cents} ${sub}`;
}

function decimal(m: string): string {
  const [a, b] = m.split('.');
  return `${a} point ${b.split('').join(' ')}`;
}

export function normalizeText(text: string): string {
  return text
    .replace(/[‘’]/g, "'")
    .replace(/«/g, '“')
    .replace(/»/g, '”')
    .replace(/[“”]/g, '"')
    .replace(/\(/g, '«')
    .replace(/\)/g, '»')
    .replace(/、/g, ', ')
    .replace(/。/g, '. ')
    .replace(/！/g, '! ')
    .replace(/，/g, ', ')
    .replace(/：/g, ': ')
    .replace(/；/g, '; ')
    .replace(/？/g, '? ')
    .replace(/[^\S \n]/g, ' ')
    .replace(/ {2,}/g, ' ')
    .replace(/(?<=\n) +(?=\n)/g, '')
    .replace(/\bD[Rr]\.(?= [A-Z])/g, 'Doctor')
    .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, 'Mister')
    .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, 'Miss')
    .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, 'Mrs')
    .replace(/\betc\.(?! [A-Z])/gi, 'etc')
    .replace(/\b(y)eah?\b/gi, "$1e'a")
    .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, splitNum)
    .replace(/(?<=\d),(?=\d)/g, '')
    .replace(/[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi, money)
    .replace(/\d*\.\d+/g, decimal)
    .replace(/(?<=\d)-(?=\d)/g, ' to ')
    .replace(/(?<=\d)S/g, ' S')
    .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
    .replace(/(?<=X')S\b/g, 's')
    .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, '-'))
    .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, '-')
    .trim();
}

const PUNCT = ';:,.!?¡¿—…"«»“”(){}[]';
const PUNCT_RE = new RegExp(`(\\s*[${PUNCT.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}]+\\s*)+`, 'g');

type Phonemize = (text: string, lang: string) => Promise<string[]>;
let phonemizePromise: Promise<Phonemize> | null = null;

async function getPhonemizer(): Promise<Phonemize> {
  phonemizePromise ??= import('phonemizer')
    .then((m) => (m as { phonemize: Phonemize }).phonemize)
    .catch(() => {
      phonemizePromise = null;
      throw new ConfigError('Kokoro needs the "phonemizer" package.', {
        hint: 'Install it: bun add phonemizer (or npm i phonemizer). It bundles espeak-ng, which is GPL-3.0 licensed.',
      });
    });
  return phonemizePromise;
}

/** Normalize, then phonemize everything outside punctuation runs. */
export async function phonemizeText(text: string, lang: 'a' | 'b'): Promise<string> {
  const phonemize = await getPhonemizer();
  const norm = normalizeText(text);
  const pieces: { match: boolean; text: string }[] = [];
  let last = 0;
  for (const m of norm.matchAll(PUNCT_RE)) {
    if (last < (m.index ?? 0)) pieces.push({ match: false, text: norm.slice(last, m.index) });
    if (m[0].length) pieces.push({ match: true, text: m[0] });
    last = (m.index ?? 0) + m[0].length;
  }
  if (last < norm.length) pieces.push({ match: false, text: norm.slice(last) });
  const espeak = lang === 'a' ? 'en-us' : 'en';
  const parts = await Promise.all(pieces.map(async (p) => (p.match ? p.text : (await phonemize(p.text, espeak)).join(' '))));
  let ph = parts
    .join('')
    .replace(/kəkˈoːɹoʊ/g, 'kˈoʊkəɹoʊ')
    .replace(/kəkˈɔːɹəʊ/g, 'kˈəʊkəɹəʊ')
    .replace(/ʲ/g, 'j')
    .replace(/r/g, 'ɹ')
    .replace(/x/g, 'k')
    .replace(/ɬ/g, 'l')
    .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, ' ')
    .replace(/ z(?=[;:,.!?¡¿—…"«»“” ]|$)/g, 'z');
  if (lang === 'a') ph = ph.replace(/(?<=nˈaɪn)ti(?!ː)/g, 'di');
  return ph.trim();
}

const ABBREV = new Set(['mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'mt', 'etc', 'vs', 'inc', 'ltd', 'co', 'e.g', 'i.e']);

/**
 * Split text into sentences. Returns complete sentences and the unfinished remainder,
 * so it can be used on a stream of text deltas.
 */
export function splitSentences(buffer: string, final = false): { sentences: string[]; rest: string } {
  const sentences: string[] = [];
  let start = 0;
  const re = /([.!?…]+["'”’)\]]*)(\s+|$)|\n{2,}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(buffer))) {
    const end = m.index + m[0].length;
    if (m[2] === '' && !final && end === buffer.length) break;
    const candidate = buffer.slice(start, m.index + (m[1]?.length ?? 0)).trim();
    const lastWord =
      candidate
        .split(/\s+/)
        .pop()
        ?.replace(/[.!?…]+$/, '')
        .toLowerCase() ?? '';
    if (m[1] === '.' && (ABBREV.has(lastWord) || /^\d+$/.test(lastWord) || /^[a-z]$/i.test(lastWord))) continue;
    if (candidate) sentences.push(candidate);
    start = end;
  }
  let rest = buffer.slice(start);
  if (final && rest.trim()) {
    sentences.push(rest.trim());
    rest = '';
  }
  return { sentences, rest };
}
