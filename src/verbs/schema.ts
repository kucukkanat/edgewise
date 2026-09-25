import { ConfigError } from '../core/errors.ts';

/** A Zod schema, or a plain JSON Schema with an optional validator. */
export type SchemaLike<T = unknown> =
  | { safeParse(v: unknown): { success: true; data: T } | { success: false; error: unknown } }
  | { jsonSchema: Record<string, unknown>; validate?: (v: unknown) => { success: true; data: T } | { success: false; error: string } };

function isZod(s: unknown): s is { safeParse: (v: unknown) => { success: boolean; data?: unknown; error?: unknown } } {
  return typeof s === 'object' && s !== null && typeof (s as { safeParse?: unknown }).safeParse === 'function';
}

/** JSON Schema for a Zod (v4) schema or a `{ jsonSchema }` object. */
export async function toJsonSchema(s: SchemaLike): Promise<Record<string, unknown>> {
  if (!isZod(s)) return (s as { jsonSchema: Record<string, unknown> }).jsonSchema;
  const own = (s as { toJSONSchema?: () => Record<string, unknown> }).toJSONSchema;
  if (typeof own === 'function') return clean(own.call(s));
  try {
    const z = (await import('zod')) as unknown as {
      toJSONSchema?: (x: unknown) => Record<string, unknown>;
      z?: { toJSONSchema?: (x: unknown) => Record<string, unknown> };
    };
    const fn = z.toJSONSchema ?? z.z?.toJSONSchema;
    if (fn) return clean(fn(s));
  } catch {
    // zod not installed
  }
  throw new ConfigError('Could not convert this schema to JSON Schema.', {
    hint: 'Use Zod 4, or pass { jsonSchema, validate } instead of a Zod 3 schema.',
  });
}

function clean(j: Record<string, unknown>): Record<string, unknown> {
  const { $schema: _drop, ...rest } = j;
  return rest;
}

function formatIssues(err: unknown): string {
  const issues = (err as { issues?: { path?: (string | number)[]; message?: string }[] })?.issues;
  if (Array.isArray(issues)) {
    return issues.map((i) => `${i.path?.length ? i.path.join('.') : '(root)'}: ${i.message}`).join('; ');
  }
  return err instanceof Error ? err.message : String(err);
}

export function validateWith<T>(s: SchemaLike<T>, value: unknown): { success: true; data: T } | { success: false; error: string; issues?: unknown } {
  if (isZod(s)) {
    const r = s.safeParse(value);
    return r.success
      ? { success: true, data: r.data as T }
      : { success: false, error: formatIssues(r.error), issues: (r.error as { issues?: unknown })?.issues };
  }
  const v = (s as { validate?: (x: unknown) => { success: true; data: T } | { success: false; error: string } }).validate;
  return v ? v(value) : { success: true, data: value as T };
}

/** Find the first complete JSON value in text (ignores prose and code fences around it). */
export function extractJson(text: string): unknown {
  const fence = /```(?:json)?\s*([\s\S]*?)```/.exec(text);
  const src = fence ? fence[1] : text;
  const start = src.search(/[[{]/);
  if (start < 0) {
    const t = src.trim();
    return JSON.parse(t);
  }
  const open = src[start];
  const close = open === '{' ? '}' : ']';
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < src.length; i++) {
    const c = src[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === '\\') esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === open || c === (open === '{' ? '[' : '{')) depth++;
    else if (c === close || c === (close === '}' ? ']' : '}')) {
      depth--;
      if (depth === 0) return JSON.parse(src.slice(start, i + 1));
    }
  }
  return JSON.parse(src.slice(start));
}

/** Best-effort parse of an incomplete JSON document, for streaming partial objects. */
export function parsePartialJson(text: string): unknown {
  const start = text.search(/[[{]/);
  if (start < 0) return undefined;
  const src = text.slice(start);
  const stack: string[] = [];
  let inStr = false;
  let esc = false;
  let lastSafe = -1;
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === '\\') esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === '{' || c === '[') stack.push(c === '{' ? '}' : ']');
    else if (c === '}' || c === ']') {
      stack.pop();
      if (!stack.length) {
        try {
          return JSON.parse(src.slice(0, i + 1));
        } catch {
          return undefined;
        }
      }
    }
    if (!inStr && (c === ',' || c === '{' || c === '[')) lastSafe = i;
  }
  const attempts = [src, lastSafe >= 0 ? src.slice(0, lastSafe + (src[lastSafe] === ',' ? 0 : 1)) : ''];
  for (const a of attempts) {
    if (!a) continue;
    let s = a;
    let q = false;
    let e = false;
    const st: string[] = [];
    for (const ch of s) {
      if (q) {
        if (e) e = false;
        else if (ch === '\\') e = true;
        else if (ch === '"') q = false;
        continue;
      }
      if (ch === '"') q = true;
      else if (ch === '{') st.push('}');
      else if (ch === '[') st.push(']');
      else if (ch === '}' || ch === ']') st.pop();
    }
    if (q) s += '"';
    s = s.replace(/[,:]\s*$/, '');
    s += st.reverse().join('');
    try {
      return JSON.parse(s);
    } catch {
      // try the next attempt
    }
  }
  return undefined;
}

export function schemaInstruction(json: Record<string, unknown>): string {
  return `Reply with only a JSON value that matches this JSON Schema, and nothing else:\n${JSON.stringify(json)}`;
}
