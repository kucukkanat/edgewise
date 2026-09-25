import { ConfigError } from '../../src/core/errors.ts';
import { mockModel } from '../../src/test/index.ts';
import { boolean, choice, evaluate, label, score, spans, stateToText } from '../../src/verbs/evaluate.ts';

const judge = mockModel({
  verb: 'evaluate',
  respond: ({ state }) => ({
    team: { billing: 0.8, tech: 0.2 },
    unsure: { a: 0.4, b: 0.35, c: 0.25 },
    severity: { low: 0.1, mid: 0.3, high: 0.6 },
    refund: state.includes('money back') ? 0.9 : 0.1,
    pii: [
      { type: 'EMAIL', start: 10, end: 25, score: 0.97 },
      { type: 'PER', start: 0, end: 3, score: 0.6 },
    ],
    kind: { question: 0.7, statement: 0.3 },
  }),
});

describe('evaluate (mock model)', () => {
  it('answers each question type', async () => {
    const { answers, confidence, info } = await evaluate({
      model: judge,
      state: 'I want my money back',
      questions: {
        team: choice({ billing: 'Charges', tech: 'Bugs' }),
        severity: score(['low', 'mid', 'high']),
        refund: boolean({ true: 'asks for a refund' }),
        kind: label(),
      },
    });
    expect(answers.team.choice).toBe('billing');
    expect(answers.team.probabilities.billing).toBeCloseTo(0.8);
    expect(answers.severity.score).toBeCloseTo(0.3 + 1.2);
    expect(answers.severity.level).toBe('high');
    expect(answers.refund.probability).toBeCloseTo(0.9);
    expect(answers.kind.label).toBe('question');
    expect(confidence.team).toBeGreaterThan(0);
    expect(confidence.team).toBeLessThan(1);
    expect(info.team.model).toBe(judge.id);
  });

  it('falls back to "otherwise" below the threshold', async () => {
    const r = await evaluate({ model: judge, state: 'x', questions: { unsure: choice({ a: 'A', b: 'B', c: 'C' }, { threshold: 0.6, otherwise: 'c' }) } });
    expect(r.answers.unsure.choice).toBe('c');
    expect(r.answers.unsure.fellBack).toBe(true);
    expect(r.answers.unsure.flagged).toBe(true);
  });

  it('flags booleans and spans over the threshold', async () => {
    const r = await evaluate({
      model: judge,
      state: 'Ana, mail ana@example.com',
      questions: { refund: boolean({ true: 'refund', threshold: 0.5 }), pii: spans(['EMAIL'], { threshold: 0.9 }) },
    });
    expect(r.answers.refund.flagged).toBe(false);
    expect(r.answers.pii.flagged).toBe(true);
    expect(r.answers.pii.spans).toHaveLength(1);
    expect(r.answers.pii.spans[0].text).toBe('ana@example.com');
  });

  it('confidence is 1 for a certain answer and 0 for an even split', async () => {
    const m = mockModel({ verb: 'evaluate', respond: () => ({ sure: { a: 1, b: 0 }, even: { a: 0.5, b: 0.5 } }) });
    const r = await evaluate({ model: m, state: 'x', questions: { sure: choice({ a: 'a', b: 'b' }), even: choice({ a: 'a', b: 'b' }) } });
    expect(r.confidence.sure).toBeCloseTo(1);
    expect(r.confidence.even).toBeCloseTo(0);
  });

  it('lets each question name its own model', async () => {
    const other = mockModel({ verb: 'evaluate', respond: () => ({ q2: 0.3 }) });
    const r = await evaluate({ model: judge, state: 'x', questions: { refund: boolean({ true: 'r' }), q2: boolean({ model: other }) } });
    expect(r.info.refund.model).toBe(judge.id);
    expect(r.info.q2.model).toBe(other.id);
  });

  it('validates questions', () => {
    expect(() => choice({ only: 'one' } as never)).toThrow(ConfigError);
    expect(() => score(['one'])).toThrow(ConfigError);
    expect(() => choice({ a: 'a', b: 'b' }, { otherwise: 'z' as never })).toThrow(ConfigError);
  });

  it('needs a model for every question', async () => {
    await expect(evaluate({ state: 'x', questions: { q: boolean() } })).rejects.toBeInstanceOf(ConfigError);
  });

  it('serializes object state deterministically', () => {
    expect(stateToText({ b: 2, a: { d: 1, c: 'x' } })).toBe(stateToText({ a: { c: 'x', d: 1 }, b: 2 }));
    expect(stateToText({ subject: 'Hi', body: 'There' })).toBe('body: There\nsubject: Hi');
    expect(stateToText('plain')).toBe('plain');
  });
});
