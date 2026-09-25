---
title: Testing your app
group: Extras
order: 4
eyebrow: edgewise/test
---

# Testing your app

`edgewise/test` makes mock models that answer instantly, so you can unit-test app logic without downloads.

```ts
import { generate, evaluate, choice } from 'edgewise';
import { mockModel } from 'edgewise/test';

const lm = mockModel({ verb: 'generate', respond: ({ messages }) => `echo: ${messages.at(-1)?.content}` });
const { text } = await generate({ model: lm, input: 'hi' }); // 'echo: hi'

const judge = mockModel({ verb: 'evaluate', respond: () => ({ topic: { billing: 0.8, other: 0.2 } }) });
const r = await evaluate({ model: judge, state: 'x', questions: { topic: choice({ billing: 'billing', other: 'other' }) } });
r.answers.topic.choice; // 'billing'
```

A mock is a real manifest, so it goes through the same resolution, Runs, streaming, schemas and tool loop as a real model. A generate mock can return tool calls:

```ts
const caller = mockModel({ verb: 'generate', respond: ({ messages }) =>
  messages.some((m) => m.role === 'tool') ? 'Done.' : { toolCalls: [{ name: 'setVolume', input: { level: 30 } }] } });
```

Mocks exist for `generate`, `evaluate`, `embed`, `speak` and `forecast`.

## Testing with real models

Edgewise's own real-model tests are in `test/models`. Run them with `bun run test:models`. They download about 5 GB the first time.
