import { z } from 'zod';
import { MarkupFilter, parsePythonCalls, parseToolCalls, tool, validateCall } from '../../src/verbs/tools.ts';

describe('tool-call parsing', () => {
  it('parses LFM2 Python-style calls', () => {
    const r = parseToolCalls('Sure.<|tool_call_start|>[setVolume(level=30), setTheme(theme="dark")]<|tool_call_end|>', 'lfm2');
    expect(r.calls).toEqual([
      { name: 'setVolume', input: { level: 30 } },
      { name: 'setTheme', input: { theme: 'dark' } },
    ]);
    expect(r.text).toBe('Sure.');
  });

  it('handles nested values, booleans, None and escapes', () => {
    expect(parsePythonCalls('[f(a=[1, 2.5, "x"], b={"k": True}, c=None, d=\'it\\\'s\')]')).toEqual([
      { name: 'f', input: { a: [1, 2.5, 'x'], b: { k: true }, c: null, d: "it's" } },
    ]);
    expect(parsePythonCalls('g()')).toEqual([{ name: 'g', input: {} }]);
    expect(parsePythonCalls('h("pos", k=-3e2)')).toEqual([{ name: 'h', input: { arg0: 'pos', k: -300 } }]);
  });

  it('accepts JSON inside LFM markers', () => {
    const r = parseToolCalls('<|tool_call_start|>[{"name": "f", "arguments": {"x": 1}}]<|tool_call_end|>', 'lfm2');
    expect(r.calls).toEqual([{ name: 'f', input: { x: 1 } }]);
  });

  it('parses Hermes (Qwen) calls', () => {
    const r = parseToolCalls(
      '<tool_call>\n{"name": "get", "arguments": {"city": "Almere"}}\n</tool_call><tool_call>{"name":"b","arguments":"{\\"n\\":2}"}</tool_call>',
      'hermes',
    );
    expect(r.calls).toEqual([
      { name: 'get', input: { city: 'Almere' } },
      { name: 'b', input: { n: 2 } },
    ]);
  });

  it('parses FunctionGemma calls', () => {
    const r = parseToolCalls(
      '<start_function_call>call:get_weather{location:<escape>London, UK<escape>,days:3,metric:true}<end_function_call>',
      'functiongemma',
    );
    expect(r.calls).toEqual([{ name: 'get_weather', input: { location: 'London, UK', days: 3, metric: true } }]);
  });

  it('returns what parsed cleanly from broken output', () => {
    expect(parsePythonCalls('[ok(a=1), broken(')).toEqual([{ name: 'ok', input: { a: 1 } }]);
    expect(parseToolCalls('<tool_call>not json</tool_call>', 'hermes').calls).toEqual([]);
  });

  it('validates arguments with the tool schema', () => {
    const tools = { setVolume: tool({ description: 'v', input: z.object({ level: z.number().int().max(100) }) }) };
    expect(validateCall(tools, { name: 'setVolume', input: { level: 30 } })).toEqual({ ok: true, input: { level: 30 } });
    const bad = validateCall(tools, { name: 'setVolume', input: { level: 300 } });
    expect(bad.ok).toBe(false);
    expect(validateCall(tools, { name: 'nope', input: {} })).toMatchObject({ ok: false, error: expect.stringContaining('Unknown tool') });
  });

  it('keeps tool markup out of streamed text, even split across chunks', () => {
    const f = new MarkupFilter('<|tool_call_start|>', '<|tool_call_end|>');
    const chunks = ['Hello ', 'there<|tool', '_call_start|>[f(a=1)]<|tool_c', 'all_end|> done'];
    const out = chunks.map((c) => f.push(c)).join('') + f.flush();
    expect(out).toBe('Hello there done');
  });

  it('requires a description and schema', () => {
    expect(() => tool({ description: '', input: z.object({}) })).toThrow();
  });
});
