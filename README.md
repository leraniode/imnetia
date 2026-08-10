<p align="left">
    <a href="https://github.com/leraniode/imnetia">
        <img src="https://raw.githubusercontent.com/leraniode/.github/main/images/imnetia.svg" />
    </a>
</p>

# Imnetia

> Deterministic intelligence framework for Go.

Built on [Centrix](https://pkg.go.devgithub.com/leraniode/x/centrix) — sparse signal mathematics.

**Status: Experimental.** APIs and behaviours are subject to change until v1.0.

---

## Purpose

Imnetia gives you the primitives to build systems that can reason, decide, and learn — deterministically, on CPU, with every conclusion fully traceable.

It is not a statistical model. It is not a wrapper around an AI provider. It is a framework for building intelligence systems from authored knowledge and sparse vector mathematics.

## Structure

```
imnetia/
  graph/      — Node, Edge, Flow (structure)
  knowledge/  — KnowledgeStore (Prototype shelf)
  runtime/    — execution engine (internal)
  learning/   — delta rule, weight updates, snapshots
  decode/     — Signal interpretation and output
  model/      — model artifact packaging
```

## License

MIT — Leraniode

---

Part of [Leraniode](https://github.com/leraniode).

<p align="left">
    <a href="https://github.com/leraniode">
        <img src="https://raw.githubusercontent.com/leraniode/.github/main/assets/footer.svg" width="600" />
    </a>
</p>
