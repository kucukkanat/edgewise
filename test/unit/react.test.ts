import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { useEvaluate } from '../../src/react/index.ts';
import { mockModel } from '../../src/test/index.ts';
import { choice } from '../../src/verbs/evaluate.ts';

// Hooks need a DOM; these run in the browser project.
const browser = typeof document !== 'undefined';
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe.skipIf(!browser)('useEvaluate', () => {
  let calls = 0;
  const judge = mockModel({
    verb: 'evaluate',
    respond: () => {
      calls++;
      return { color: { red: 0.9, blue: 0.1 } };
    },
  });
  const questions = { color: choice({ red: 'Red', blue: 'Blue' }) };
  const px = () => ({ data: new Uint8Array(12), width: 2, height: 2, channels: 3 as const });
  type Out = ReturnType<typeof useEvaluate<typeof questions>>;

  async function mount() {
    const root = createRoot(document.createElement('div'));
    let out: Out | undefined;
    const Probe = (p: Parameters<typeof useEvaluate<typeof questions>>[0]) => {
      out = useEvaluate(p);
      return null;
    };
    const render = async (p: Parameters<typeof useEvaluate<typeof questions>>[0]) => {
      await act(async () => root.render(createElement(Probe, p)));
      await act(() => new Promise((r) => setTimeout(r, 30)));
    };
    return { render, out: () => out, unmount: () => act(() => root.unmount()) };
  }

  beforeEach(() => {
    calls = 0;
  });

  it('runs on images alone', async () => {
    const h = await mount();
    await h.render({ model: judge, images: [px()], questions, debounceMs: 0 });
    expect(h.out()?.answers?.color.choice).toBe('red');
    expect(calls).toBe(1);
    await h.unmount();
  });

  it('waits for a state or media', async () => {
    const h = await mount();
    await h.render({ model: judge, state: '', questions, debounceMs: 0 });
    expect(calls).toBe(0);
    expect(h.out()?.answers).toBeNull();
    await h.unmount();
  });

  it('re-runs when the images change, not when the array around them does', async () => {
    const h = await mount();
    const img = px();
    await h.render({ model: judge, images: [img], questions, debounceMs: 0 });
    await h.render({ model: judge, images: [img], questions, debounceMs: 0 });
    expect(calls).toBe(1);
    await h.render({ model: judge, images: [px()], questions, debounceMs: 0 });
    expect(calls).toBe(2);
    const clip = new Float32Array(16000);
    await h.render({ model: judge, audio: clip, questions, debounceMs: 0 });
    await h.render({ model: judge, audio: clip, questions, debounceMs: 0 });
    expect(calls).toBe(3);
    await h.unmount();
  });

  it('re-runs when the state changes', async () => {
    const h = await mount();
    await h.render({ model: judge, state: { a: 1 }, questions, debounceMs: 0 });
    await h.render({ model: judge, state: { a: 1 }, questions, debounceMs: 0 });
    await h.render({ model: judge, state: { a: 2 }, questions, debounceMs: 0 });
    expect(calls).toBe(2);
    await h.unmount();
  });
});
