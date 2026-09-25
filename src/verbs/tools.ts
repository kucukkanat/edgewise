import { ConfigError } from '../core/errors.ts';
import { type SchemaLike, toJsonSchema, validateWith } from './schema.ts';

export interface Tool<I = unknown, O = unknown> {
  description: string;
  input: SchemaLike<I>;
  execute?: (input: I, ctx: { signal?: AbortSignal; call: ToolCallRef }) => Promise<O> | O;
}

export interface ToolCallRef {
  id: string;
  name: string;
}

/** Define a tool the model can call. `input` is a Zod schema (or any object with `safeParse` and a JSON Schema). */
export function tool<I, O>(def: Tool<I, O>): Tool<I, O> {
  if (!def.description) throw new ConfigError('tool() needs a description.');
  if (!def.input) throw new ConfigError('tool() needs an input schema.');
  return def;
}

// biome-ignore lint/suspicious/noExplicitAny: tools are heterogeneous by design
export type AnyTool = Tool<any, any>;

export type ToolFormat = 'lfm2' | 'hermes' | 'functiongemma';

/** JSON-Schema tool list in the OpenAI-style shape chat templates expect. */
export async function toolsForTemplate(tools: Record<string, AnyTool>): Promise<unknown[]> {
  const out: unknown[] = [];
  for (const [name, t] of Object.entries(tools)) {
    out.push({
      type: 'function',
      function: { name, description: t.description, parameters: await toJsonSchema(t.input) },
    });
  }
  return out;
}

export interface RawCall {
  name: string;
  input: unknown;
}

const MARKERS: Record<ToolFormat, { start: string; end: string }> = {
  lfm2: { start: '<|tool_call_start|>', end: '<|tool_call_end|>' },
  hermes: { start: '<tool_call>', end: '</tool_call>' },
  functiongemma: { start: '<start_function_call>', end: '<end_function_call>' },
};

export function toolMarkers(format: ToolFormat): { start: string; end: string } {
  return MARKERS[format];
}

/** Extract tool calls from raw model output. Returns the calls and the text outside them. */
export function parseToolCalls(raw: string, format: ToolFormat): { calls: RawCall[]; text: string } {
  const { start, end } = MARKERS[format];
  const calls: RawCall[] = [];
  let text = '';
  let i = 0;
  while (i < raw.length) {
    const s = raw.indexOf(start, i);
    if (s < 0) {
      text += raw.slice(i);
      break;
    }
    text += raw.slice(i, s);
    const e = raw.indexOf(end, s + start.length);
    const body = raw.slice(s + start.length, e < 0 ? raw.length : e).trim();
    i = e < 0 ? raw.length : e + end.length;
    if (!body) continue;
    if (format === 'lfm2') calls.push(...parseLfmBody(body));
    else if (format === 'hermes') calls.push(...parseHermesBody(body));
    else calls.push(...parseFunctionGemmaBody(body));
  }
  return { calls, text: text.trim() };
}

function parseLfmBody(body: string): RawCall[] {
  const t = body.trim();
  // Some LFM checkpoints emit JSON instead of Python syntax.
  if (t.startsWith('{') || (t.startsWith('[') && /^\[\s*\{/.test(t))) {
    try {
      return jsonCalls(JSON.parse(t));
    } catch {
      // fall through to the Python parser
    }
  }
  return parsePythonCalls(t);
}

function parseHermesBody(body: string): RawCall[] {
  try {
    return jsonCalls(JSON.parse(body));
  } catch {
    return [];
  }
}

function jsonCalls(v: unknown): RawCall[] {
  const arr = Array.isArray(v) ? v : [v];
  const out: RawCall[] = [];
  for (const c of arr) {
    if (c && typeof c === 'object') {
      const o = c as { name?: string; arguments?: unknown; parameters?: unknown; function?: { name?: string; arguments?: unknown } };
      const name = o.name ?? o.function?.name;
      let input = o.arguments ?? o.parameters ?? o.function?.arguments ?? {};
      if (typeof input === 'string') {
        try {
          input = JSON.parse(input);
        } catch {
          // keep as string
        }
      }
      if (name) out.push({ name, input });
    }
  }
  return out;
}

// ------------------------------------------------ FunctionGemma: call:name{key:value,...}

function parseFunctionGemmaBody(body: string): RawCall[] {
  const out: RawCall[] = [];
  const re = /call:([A-Za-z_][\w.-]*)\s*\{/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body))) {
    const name = m[1];
    let i = re.lastIndex;
    const input: Record<string, unknown> = {};
    const esc = '<escape>';
    while (i < body.length && body[i] !== '}') {
      while (body[i] === ',' || body[i] === ' ') i++;
      const colon = body.indexOf(':', i);
      if (colon < 0) break;
      const key = body.slice(i, colon).trim();
      i = colon + 1;
      let value: unknown;
      if (body.startsWith(esc, i)) {
        const close = body.indexOf(esc, i + esc.length);
        value = body.slice(i + esc.length, close < 0 ? body.length : close);
        i = close < 0 ? body.length : close + esc.length;
      } else {
        let j = i;
        let depth = 0;
        while (j < body.length && !(depth === 0 && (body[j] === ',' || body[j] === '}'))) {
          if (body[j] === '[' || body[j] === '{') depth++;
          if (body[j] === ']' || body[j] === '}') depth--;
          j++;
        }
        const rawV = body.slice(i, j).trim();
        value = rawV === 'true' ? true : rawV === 'false' ? false : rawV !== '' && !Number.isNaN(Number(rawV)) ? Number(rawV) : rawV;
        i = j;
      }
      if (key) input[key] = value;
    }
    re.lastIndex = i;
    out.push({ name, input });
  }
  return out;
}

// ------------------------------------------------ Python call syntax: [f(a=1, b="x"), g()]

class PyParser {
  i = 0;
  constructor(private s: string) {}
  ws() {
    while (this.i < this.s.length && /\s/.test(this.s[this.i])) this.i++;
  }
  peek() {
    this.ws();
    return this.s[this.i];
  }
  eat(ch: string) {
    this.ws();
    if (this.s[this.i] !== ch) throw new Error(`expected ${ch} at ${this.i}`);
    this.i++;
  }
  ident(): string {
    this.ws();
    const m = /^[A-Za-z_][\w.-]*/.exec(this.s.slice(this.i));
    if (!m) throw new Error(`expected identifier at ${this.i}`);
    this.i += m[0].length;
    return m[0];
  }
  value(): unknown {
    const c = this.peek();
    if (c === '"' || c === "'") return this.str();
    if (c === '[' || c === '(') {
      const close = c === '[' ? ']' : ')';
      this.i++;
      const arr: unknown[] = [];
      while (this.peek() !== close) {
        arr.push(this.value());
        if (this.peek() === ',') this.i++;
      }
      this.i++;
      return arr;
    }
    if (c === '{') {
      this.i++;
      const obj: Record<string, unknown> = {};
      while (this.peek() !== '}') {
        const k = this.value();
        this.eat(':');
        obj[String(k)] = this.value();
        if (this.peek() === ',') this.i++;
      }
      this.i++;
      return obj;
    }
    const m = /^-?\d+(\.\d+)?([eE][-+]?\d+)?/.exec(this.s.slice(this.i));
    if (m) {
      this.i += m[0].length;
      return Number(m[0]);
    }
    const id = this.ident();
    if (id === 'True' || id === 'true') return true;
    if (id === 'False' || id === 'false') return false;
    if (id === 'None' || id === 'null') return null;
    return id;
  }
  str(): string {
    this.ws();
    const q = this.s[this.i++];
    let out = '';
    while (this.i < this.s.length && this.s[this.i] !== q) {
      if (this.s[this.i] === '\\') {
        const n = this.s[this.i + 1];
        out += n === 'n' ? '\n' : n === 't' ? '\t' : n;
        this.i += 2;
      } else out += this.s[this.i++];
    }
    this.i++;
    return out;
  }
  call(): RawCall {
    const name = this.ident();
    this.eat('(');
    const input: Record<string, unknown> = {};
    let pos = 0;
    while (this.peek() !== ')') {
      const save = this.i;
      let key: string | null = null;
      try {
        const id = this.ident();
        if (this.peek() === '=') {
          this.i++;
          key = id;
        } else this.i = save;
      } catch {
        this.i = save;
      }
      const v = this.value();
      input[key ?? `arg${pos++}`] = v;
      if (this.peek() === ',') this.i++;
    }
    this.i++;
    return { name, input };
  }
}

export function parsePythonCalls(s: string): RawCall[] {
  const p = new PyParser(s);
  const out: RawCall[] = [];
  try {
    const bracket = p.peek() === '[';
    if (bracket) p.i++;
    while (p.i < s.length) {
      const c = p.peek();
      if (c === undefined || c === ']') break;
      out.push(p.call());
      if (p.peek() === ',') p.i++;
    }
  } catch {
    // Return what parsed cleanly.
  }
  return out;
}

/** Validate a raw call against the tool's schema. */
export function validateCall(tools: Record<string, AnyTool>, call: RawCall): { ok: true; input: unknown } | { ok: false; error: string } {
  const t = tools[call.name];
  if (!t) return { ok: false, error: `Unknown tool "${call.name}". Available: ${Object.keys(tools).join(', ')}.` };
  const r = validateWith(t.input, call.input);
  return r.success ? { ok: true, input: r.data } : { ok: false, error: r.error };
}

/**
 * Filter streamed text so tool-call markup never reaches the user.
 * Holds back a possible partial marker at the end of each chunk.
 */
export class MarkupFilter {
  private buf = '';
  private inside = false;
  constructor(
    private readonly start: string,
    private readonly end: string,
  ) {}
  push(chunk: string): string {
    this.buf += chunk;
    let out = '';
    for (;;) {
      if (this.inside) {
        const e = this.buf.indexOf(this.end);
        if (e < 0) {
          this.buf = this.buf.slice(Math.max(0, this.buf.length - this.end.length));
          return out;
        }
        this.buf = this.buf.slice(e + this.end.length);
        this.inside = false;
      } else {
        const s = this.buf.indexOf(this.start);
        if (s >= 0) {
          out += this.buf.slice(0, s);
          this.buf = this.buf.slice(s + this.start.length);
          this.inside = true;
          continue;
        }
        let keep = 0;
        for (let k = Math.min(this.start.length - 1, this.buf.length); k > 0; k--) {
          if (this.start.startsWith(this.buf.slice(-k))) {
            keep = k;
            break;
          }
        }
        out += this.buf.slice(0, this.buf.length - keep);
        this.buf = this.buf.slice(this.buf.length - keep);
        return out;
      }
    }
  }
  flush(): string {
    const r = this.inside ? '' : this.buf;
    this.buf = '';
    return r;
  }
}
