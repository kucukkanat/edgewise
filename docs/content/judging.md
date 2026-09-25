---
title: Routing, linting and redaction
group: Guides
order: 4
---

# Routing, linting and redaction

Three jobs that `evaluate` does well, with helpers in `edgewise/helpers`.

## Route prompts

Send each prompt to the right place: a bigger model, a tool, a human.

```ts
import { route } from 'edgewise/helpers';

const lanes = { coding: 'Programming help', billing: 'Payments and invoices', chat: 'Small talk' };
const r = await route(userText, lanes, { model: 'lfm2.5-encoder-350m-router', threshold: 0.5, otherwise: 'chat' });
r.route;         // 'coding'
r.probabilities; // { coding: 0.84, billing: 0.06, chat: 0.1 }
r.fellBack;      // true when the top lane was below the threshold
```

The default model is `judge:router` (a zero-shot NLI model). The LFM2.5 prompt router is trained for this job and is multilingual.

## Lint against policies

A policy is a question with a threshold. `lint` flags every answer that reaches it:

```ts
import { boolean, spans } from 'edgewise';
import { lint } from 'edgewise/helpers';

const { passed, findings } = await lint(draft, {
  injection: boolean({ model: 'judge:injection', threshold: 0.9 }),
  personalData: spans(['contact.email', 'contact.phone'], { model: 'judge:pii', threshold: 0.8 }),
  offTopic: boolean({ model: 'judge:router', true: 'not about our product', threshold: 0.7 }),
});
if (!passed) console.log(findings.map((f) => f.policy)); // ['injection']
```

Policies without a threshold are refused, so a lint never passes by accident.

## Redact personal data

```ts
import { redact } from 'edgewise/helpers';

await redact('Email Dr. Laura Schmidt at laura@charite.de'); // judge:pii, the LFM2.5 PII encoder
// 'Email [IDENTITY.PERSON_NAME] at [CONTACT.EMAIL]'

await redact(text, { types: ['contact.email'], mask: (s) => '█'.repeat(s.end - s.start) });
```

`replaceSpans(text, spans, mask)` does the replacement for spans you already have.

> [!WARN] Check model licences before you ship. `piiranha-v1` is CC BY-NC-ND 4.0, which forbids commercial use. The LFM2.5 encoders use the LFM Open License, which has conditions for large companies. Restrict what can load with `configure({ licenses: [...] })`.
