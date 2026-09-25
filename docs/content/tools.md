---
title: Tool calling
group: Guides
order: 3
---

# Tool calling

## Define tools

```ts
import { generate, tool } from 'edgewise';
import { z } from 'zod';

const tools = {
  getWeather: tool({
    description: 'Current weather for a city',
    input: z.object({ city: z.string() }),
    execute: async ({ city }) => fetchWeather(city),
  }),
  addTodo: tool({
    description: 'Add an item to the to-do list',
    input: z.object({ title: z.string(), due: z.string().optional() }),
    execute: async (todo) => db.todos.add(todo),
  }),
};

const { text, toolCalls, toolResults } = await generate({ model: 'text:default', input: 'Remind me to call Ana tomorrow', tools });
```

## The loop

When the model calls tools, Edgewise validates each call's input with its schema, runs `execute`, adds the results to the conversation, and calls the model again, up to `maxSteps` (default 3). An invalid call is sent back to the model as an error instead of being run. The run ends when the model answers in text, or with `finishReason: 'tool-calls'` when steps run out.

Tools without `execute` are returned in `toolCalls` for you to handle.

## Approve before running

```ts
await generate({
  model: 'text:default',
  input,
  tools,
  approve: async (call) => call.name !== 'deleteAll' && confirm(`Run ${call.name}?`),
});
```

A declined call is reported to the model as declined.

## Events

```ts
const run = generate({ model: 'text:default', input, tools });
for await (const e of run.events) {
  if (e.type === 'tool-call') log(`→ ${e.call.name}`, e.call.input);
  if (e.type === 'tool-result') log(`← ${e.result.name}`, e.result.output);
}
```

## Models and formats

Each model family writes tool calls its own way. Edgewise speaks each format:

| Model | Format |
| --- | --- |
| LFM2 and LFM2.5 | `<|tool_call_start|>[fn(a=1)]<|tool_call_end|>`, Python-style calls |
| Qwen3 | Hermes: `<tool_call>{"name": …}</tool_call>` |
| FunctionGemma | `<start_function_call>call:fn{…}<end_function_call>` |

Small models call tools best when the tool names and descriptions are plain and specific, and when there are fewer than ten tools. `functiongemma-270m` and `lfm2-1.2b-tool` are trained for app actions.
