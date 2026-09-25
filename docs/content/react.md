---
title: React hooks
group: Extras
order: 2
eyebrow: edgewise/react
---

# React hooks

```bash
bun add edgewise react
```

## Provider

`EdgewiseProvider` applies configuration once. It is optional.

```tsx
import { EdgewiseProvider } from 'edgewise/react';

<EdgewiseProvider allowPreview maxLoadedModels={2}>
  <App />
</EdgewiseProvider>
```

## Load a model with progress

```tsx
import { useModel } from 'edgewise/react';

function ModelGate({ children }) {
  const m = useModel('text:default');
  if (m.status === 'unsupported') return <p>This device cannot run the model.</p>;
  if (m.status !== 'ready')
    return <button onClick={m.load}>{m.status === 'idle' ? `Download ${Math.round((m.sizeBytes ?? 0) / 1e6)} MB` : `${Math.round(m.progress * 100)}%`}</button>;
  return children;
}
```

`status` is `idle`, `downloading`, `compiling`, `ready`, `error` or `unsupported`. `device` says where it will run.

## Chat

```tsx
import { useChat } from 'edgewise/react';

function Chat() {
  const { messages, input, setInput, submit, isStreaming, stop } = useChat({ model: 'vision:default' });
  return (
    <form onSubmit={(e) => (e.preventDefault(), submit())}>
      {messages.map((m) => <p key={m.id}><b>{m.role}</b> {typeof m.content === 'string' ? m.content : ''}</p>)}
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      {isStreaming ? <button type="button" onClick={stop}>Stop</button> : <button>Send</button>}
    </form>
  );
}
```

`attach(blob)` adds an image or audio file to the next message when the model accepts it.

## The other hooks

| Hook | Returns |
| --- | --- |
| `useGenerate(opts)` | `{ text, result, isRunning, error, run(input?), stop }` |
| `useEvaluate({ state, questions, debounceMs })` | `{ answers, confidence, pending, error }`, re-run as `state` changes |
| `useEmbed({ model })` | `{ embed(text), pending }` |
| `useSpeak({ model, voice })` | `{ speak(text), speaking, stop }` |
| `usePaint({ model, size, steps })` | `{ run(prompt), result, preview, isRunning, stop }` |
| `useForecast({ model, series, horizon })` | `{ forecast, error }` |
| `useMic(opts)` | `{ start, stop, listening, level, error, source }` |
| `useCapabilities()` | `Capabilities` or `null` while loading |
