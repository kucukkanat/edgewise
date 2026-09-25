import { edgewise, edgewiseEvaluationModel, edgewiseLanguageModel, fromPrompt, toAiSdkQuestions } from '../../src/ai-sdk/index.ts';
import { mockModel } from '../../src/test/index.ts';
import { boolean, choice, score, spans } from '../../src/verbs/evaluate.ts';

describe('Vercel AI SDK adapter', () => {
  const lm = mockModel({ verb: 'generate', respond: () => 'Hi there' });

  it('converts AI SDK prompts', () => {
    const msgs = fromPrompt([
      { role: 'system', content: 'sys' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'hello' },
          { type: 'file', mediaType: 'image/png', data: new Uint8Array([1, 2]) as never },
        ],
      },
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'f', input: { a: 1 } }] },
      { role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'f', output: { type: 'json', value: { ok: true } } }] },
    ]);
    expect(msgs[0]).toEqual({ role: 'system', content: 'sys' });
    expect((msgs[1].content as { type: string }[]).map((p) => p.type)).toEqual(['text', 'image']);
    expect(msgs[2].content).toContain('"name":"f"');
    expect(msgs[3]).toMatchObject({ role: 'tool', toolCallId: 'c1', content: '{"ok":true}' });
  });

  it('implements doGenerate and doStream', async () => {
    const m = edgewiseLanguageModel(lm);
    expect(m.specificationVersion).toBe('v4');
    const r = await m.doGenerate({ prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] });
    expect(r.content).toEqual([{ type: 'text', text: 'Hi there' }]);
    expect(r.finishReason.unified).toBe('stop');
    const { stream } = await m.doStream({ prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] });
    const parts: string[] = [];
    let text = '';
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value.type);
      if (value.type === 'text-delta') text += value.delta;
    }
    expect(text).toBe('Hi there');
    expect(parts[0]).toBe('stream-start');
    expect(parts.at(-1)).toBe('finish');
  });

  it('returns tool calls for the AI SDK to execute', async () => {
    const tm = mockModel({ verb: 'generate', respond: () => ({ toolCalls: [{ name: 'weather', input: { city: 'Almere' } }] }) });
    const r = await edgewiseLanguageModel(tm).doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'weather?' }] }],
      tools: [{ type: 'function', name: 'weather', description: 'Get weather', inputSchema: { type: 'object', properties: { city: { type: 'string' } } } }],
    });
    expect(r.finishReason.unified).toBe('tool-calls');
    expect(r.content).toEqual([{ type: 'tool-call', toolCallId: 'call_1', toolName: 'weather', input: '{"city":"Almere"}' }]);
  });

  it('exposes evaluate models to experimental_evaluate', async () => {
    const judge = mockModel({ verb: 'evaluate', respond: () => ({ team: { billing: 0.9, tech: 0.1 }, refund: 0.8 }) });
    const r = await edgewiseEvaluationModel(judge).doEvaluate({
      state: 'x',
      questions: {
        team: { type: 'choice', instructions: 'Which team?', criteria: { billing: 'Charges', tech: null } },
        refund: { type: 'boolean', instructions: 'Asks for a refund?' },
      },
    });
    expect(r.answers.team).toMatchObject({ type: 'choice', choice: 'billing' });
    expect(r.answers.refund).toMatchObject({ type: 'boolean', probability: 0.8 });
  });

  it('picks the adapter by verb', () => {
    expect((edgewise('embed:tiny') as { doEmbed?: unknown }).doEmbed).toBeTypeOf('function');
    expect(() => edgewise('kokoro-82m')).toThrow(/adapter covers/);
  });

  it('converts questions for cloud judges', () => {
    const q = toAiSdkQuestions({
      department: choice({ billing: 'Charges', tech: 'Bugs' }, { instructions: 'Which team handles this?' }),
      severity: score(['low', 'high']),
      wantsRefund: boolean({ true: 'Asks for money back' }),
    });
    expect(q.department).toEqual({ type: 'choice', instructions: 'Which team handles this?', criteria: { billing: 'Charges', tech: 'Bugs' } });
    expect(q.severity).toEqual({ type: 'score', instructions: 'severity', criteria: ['low', 'high'] });
    expect(q.wantsRefund).toMatchObject({ type: 'boolean', instructions: 'wants Refund' });
    expect(() => toAiSdkQuestions({ p: spans() })).toThrow(/no AI SDK equivalent/);
  });
});
