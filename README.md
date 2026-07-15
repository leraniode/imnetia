# Imnetia

> Deterministic intelligence framework for Go.

Built on [Centrix](https://github.com/leraniode/xgo/centrix) — sparse signal mathematics.

**Status: Experimental.** APIs and behaviours are subject to change until v1.0.

---

## What it is

Imnetia gives you the primitives to build systems that can reason, decide, and learn — deterministically, on CPU, with every conclusion fully traceable.

It is not a statistical model. It is not a wrapper around an AI provider. It is a framework for building intelligence systems from authored knowledge and sparse vector mathematics.

## Organisation

```
imnetia/
  graph/      — Node, Edge, Flow (structure)
  knowledge/  — KnowledgeStore (Prototype shelf)
  runtime/    — execution engine (internal)
  learning/   — delta rule, weight updates, snapshots
  decode/     — Signal interpretation and output
  model/      — model artifact packaging
```

## Part of Leraniode

[leraniode](https://github.com/leraniode)
