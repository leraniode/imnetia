package engine

import (
	"errors"

	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/internal/runtime"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
	"github.com/leraniode/x/centrix/registry"
)

// ErrNoStore is returned when Run is called on an Engine with no KnowledgeStore.
var ErrNoStore = errors.New("imnetia/engine: Engine has no KnowledgeStore")

// Engine is the bridge between the developer's Flow and Context
// and Imnetia's internal execution runtime.
//
// The Engine owns the KnowledgeStore, Registry, and internal executor.
// Its public surface is intentionally small.
//
// Engine is safe for concurrent use — multiple goroutines may call
// Run simultaneously. The executor is stateless per-run.
type Engine struct {
	store    *knowledge.KnowledgeStore
	reg      *registry.Registry
	executor *runtime.Executor
}

// New creates an Engine backed by the given KnowledgeStore and Registry.
//
// The Registry is used by the ContextEngine to encode raw text queries
// (Context key "query") into SparseVectors before the flow runs.
// Pass nil for reg if you only use pre-encoded vectors (Context key "input").
//
// The store should be fully loaded before calling Run.
func New(store *knowledge.KnowledgeStore, reg *registry.Registry) *Engine {
	if reg == nil {
		reg = registry.New() // empty registry — pre-encoded input only
	}
	return &Engine{
		store:    store,
		reg:      reg,
		executor: runtime.NewExecutor(store, reg),
	}
}

// Run executes the Flow from its entry node and returns the final Signal.
//
// The final Signal carries:
//   - Vector: accumulated feature state after all node transformations
//   - Confidence: certainty score [0.0, 1.0]
//   - Trace: complete execution log
//
// Input paths (checked in priority order):
//   - ctx.Set("input", core.SparseVector{...}) — pre-encoded vector
//   - ctx.Set("query", "raw text here")        — auto-encoded via Registry
//
// The caller interprets the Signal via imnetia/decode.
func (e *Engine) Run(flow *graph.Flow, ctx Context) (core.Signal, error) {
	if e.store == nil {
		return core.Signal{}, ErrNoStore
	}
	return e.executor.Execute(flow, ctx)
}

// Store returns the Engine's KnowledgeStore.
// Used by imnetia/learning to update Prototype weights after a run.
func (e *Engine) Store() *knowledge.KnowledgeStore {
	return e.store
}

// Registry returns the Engine's Registry.
// Used by nodes (via Context) for feature name lookups and debugging.
func (e *Engine) Registry() *registry.Registry {
	return e.reg
}
