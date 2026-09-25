---
title: evaluate
group: The six verbs
order: 20
eyebrow: verb · → decisions
---

# evaluate

Ask small encoder models typed questions about a piece of text. Answers come back as probabilities in milliseconds, with no text generation. Use it for routing, moderation, policy checks, PII and entity extraction.

## Ask questions

```ts
import { boolean, choice, evaluate, score, spans } from 'edgewise';

const { answers, confidence } = await evaluate({
  model: 'judge:router',
  state: 'My package arrived broken and I want my money back.',
  questions: {
    topic: choice({ refund: 'a refund request', billing: 'a billing question', praise: 'praise' }),
    mood: score(['calm', 'annoyed', 'furious']),
    wantsRefund: boolean({ true: 'wants a refund' }),
  },
});

answers.topic.choice;          // 'refund'
answers.topic.probabilities;   // { refund: 0.93, billing: 0.05, praise: 0.02 }
answers.mood.level;            // 'annoyed'
answers.wantsRefund.probability; // 0.97
confidence.topic;              // 0..1
```

`state` is a string, or any JSON value. Objects are serialized with sorted keys so the same state always gives the same answer.

## Question types

| Builder | Answer |
| --- | --- |
| `choice({ key: 'description', … }, { otherwise, threshold })` | `choice`, `probabilities`, `fellBack` |
| `score(['low', 'mid', 'high'])` | `score` (0..1), `level`, `probabilities` |
| `boolean({ true: 'description' })` | `probability` |
| `spans(['EMAIL', …])` | `spans: { type, start, end, text, score }[]` |
| `label()` | `label`, `probabilities` from a fixed-label classifier |

Answers are typed from the question: `answers.topic.choice` has type `'refund' | 'billing' | 'praise'`.

Every question can set its own `model`, `threshold` and `instructions`. With a `threshold`, answers carry `flagged`: booleans, scores and spans are flagged when they reach it, and a choice falls back to `otherwise` (and is flagged) when its top probability is below it.

## How it works

Edgewise picks the method from the model:

- **NLI models** (`nli-deberta-v3-xsmall`) score each description as a hypothesis against the text: zero-shot, any labels.
- **The LFM2.5 prompt router** (`lfm2.5-encoder-350m-router`) embeds the text and each category description, and compares them. It is trained for routing and is multilingual.
- **Classifiers** (`prompt-injection-deberta-v3`) have fixed labels. `boolean()` maps to the positive label; `label()` returns all of them.
- **Token classifiers** (`bert-base-ner`, `lfm2.5-encoder-350m-pii`, `piiranha-v1`) answer `spans()`, with character offsets into your text.

A question the model cannot answer throws `UnsupportedInputError` naming models that can.

## Calibrate thresholds

Zero-shot probabilities are relative, not absolute. Before you gate on a threshold, run 50 to 100 labelled examples from your own traffic and pick the value that gives the precision you need. `confidence` (1 minus normalized entropy) helps spot inputs where the model is unsure.

## Models

| Model | Answers | Notes |
| --- | --- | --- |
| `nli-deberta-v3-xsmall` | choice, score, boolean | alias `judge:router`; zero-shot, English |
| `lfm2.5-encoder-350m-router` | choice, score, boolean | LiquidAI prompt router, multilingual |
| `prompt-injection-deberta-v3` | boolean, label | alias `judge:injection` |
| `bert-base-ner` | spans | alias `judge:entities`; PER, ORG, LOC, MISC |
| `lfm2.5-encoder-350m-pii` | spans | alias `judge:pii`; LiquidAI PII detector, fine-grained types |
| `piiranha-v1` | spans | preview; licence cc-by-nc-nd-4.0 (non-commercial) |

For the full, current list see [Models](models).
