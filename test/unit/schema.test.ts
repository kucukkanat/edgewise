import { z } from 'zod';
import { extractJson, parsePartialJson, toJsonSchema, validateWith } from '../../src/verbs/schema.ts';

describe('structured output helpers', () => {
  it('extracts JSON from prose and code fences', () => {
    expect(extractJson('Here you go: {"a": 1, "b": [2, 3]} hope that helps')).toEqual({ a: 1, b: [2, 3] });
    expect(extractJson('```json\n{"x": "}"}\n```')).toEqual({ x: '}' });
    expect(extractJson('[1, {"a": 2}]')).toEqual([1, { a: 2 }]);
    expect(() => extractJson('no json')).toThrow();
  });

  it('parses partial JSON while it streams', () => {
    expect(parsePartialJson('{"who": "An')).toEqual({ who: 'An' });
    expect(parsePartialJson('{"a": 1, "b": [1, 2')).toEqual({ a: 1, b: [1, 2] });
    expect(parsePartialJson('{"a": 1,')).toEqual({ a: 1 });
    expect(parsePartialJson('nothing yet')).toBeUndefined();
  });

  it('converts Zod 4 schemas to JSON Schema', async () => {
    const js = await toJsonSchema(z.object({ name: z.string(), age: z.number().optional() }));
    expect(js.type).toBe('object');
    expect(js.properties).toHaveProperty('name');
    expect(js.required).toEqual(['name']);
    expect(js).not.toHaveProperty('$schema');
  });

  it('accepts plain JSON Schema with a custom validator', async () => {
    const s = {
      jsonSchema: { type: 'number' },
      validate: (v: unknown) => (typeof v === 'number' ? { success: true as const, data: v } : { success: false as const, error: 'not a number' }),
    };
    expect(await toJsonSchema(s)).toEqual({ type: 'number' });
    expect(validateWith(s, 3)).toEqual({ success: true, data: 3 });
    expect(validateWith(s, 'x')).toMatchObject({ success: false, error: 'not a number' });
  });

  it('reports Zod issues with paths', () => {
    const r = validateWith(z.object({ a: z.object({ b: z.number() }) }), { a: { b: 'x' } });
    expect(r.success).toBe(false);
    if (!r.success) expect(r.error).toContain('a.b');
  });
});
