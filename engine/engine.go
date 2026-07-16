package engine

import (
	"errors"

	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/internal/runtime"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

// ErrNoStore is returned when Run is called on an Engine with no KnowledgeStore.
var ErrNoStore = errors.New("imnetia/engine: Engine has no KnowledgeStore")

// Engine is the bridge between the developer's Flow and Context
// and Imnetia's internal execution runtime.
//
// The Engine owns the KnowledgeStore and the internal executor.
// Its public surface is intentionally small.
//
// Engine is safe for concurrent use — multiple goroutines may call
// Run simultaneously. The executor is stateless per-run.
type Engine struct {
	store    *knowledge.KnowledgeStore
	executor *runtime.Executor
}

// New creates an Engine backed by the given KnowledgeStore.
// The store should be fully loaded before calling Run.
func New(store *knowledge.KnowledgeStore) *Engine {
	return &Engine{
		store:    store,
		executor: runtime.NewExecutor(store),
	}
}

// Run executes the Flow from its entry node and returns the final Signal.
//
// The final Signal carries:
//   - Vector: accumulated feature state after all node transformations
//   - Confidence: certainty score [0.0, 1.0]
//   - Trace: complete execution log
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
