package runtime

import (
	"errors"
	"fmt"

	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/imnetia/types"
	"github.com/leraniode/xgo/centrix/core"
)

const maxVisits = 50

// ErrCycleDetected is returned when a node exceeds maxVisits in one run.
var ErrCycleDetected = errors.New("imnetia/runtime: cycle detected — node visited too many times")

// runResult carries both outputs of a NodeHandler invocation.
type runResult struct {
	signal core.Signal
	next   types.NodeResult
}

// Executor runs a Flow to completion, producing a final Signal.
// Stateless — safe to reuse across concurrent runs.
type Executor struct {
	ke *knowledgeEngine
	ce *contextEngine
}

// NewExecutor creates an Executor backed by the given KnowledgeStore.
func NewExecutor(store *knowledge.KnowledgeStore) *Executor {
	return &Executor{
		ke: newKnowledgeEngine(store),
		ce: newContextEngine(),
	}
}

// Execute runs the flow and returns the final Signal.
func (ex *Executor) Execute(f *graph.Flow, ctx types.Context) (core.Signal, error) {
	if err := f.Validate(); err != nil {
		return core.Signal{}, err
	}

	signal := ex.ce.Project(ctx)
	visits := make(map[string]int)
	currentID := f.EntryID()

	for currentID != "" {
		visits[currentID]++
		if visits[currentID] > maxVisits {
			return core.Signal{}, fmt.Errorf("%w: %q", ErrCycleDetected, currentID)
		}

		node, ok := f.Node(currentID)
		if !ok {
			return core.Signal{}, fmt.Errorf("%w: %q", graph.ErrNodeNotFound, currentID)
		}

		handler, ok := node.Handler.(types.NodeHandler)
		if !ok {
			return core.Signal{}, fmt.Errorf(
				"imnetia/runtime: node %q handler is not types.NodeHandler", currentID,
			)
		}

		// Run node in its own goroutine.
		resultCh := make(chan runResult, 1)
		go func(h types.NodeHandler, s core.Signal, c types.Context) {
			sig, nr := h(c, s)
			resultCh <- runResult{signal: sig, next: nr}
		}(handler, signal, ctx)

		r := <-resultCh
		signal = r.signal
		currentID = ex.nextNode(f, currentID, r.next)
	}

	return signal, nil
}

// nextNode determines the next node to visit.
// Priority: NodeResult.Next → highest-weight edge → "" (terminate).
func (ex *Executor) nextNode(f *graph.Flow, currentID string, nr types.NodeResult) string {
	if nr.Next != "" {
		if _, ok := f.Node(nr.Next); ok {
			return nr.Next
		}
	}

	edges := f.OutEdges(currentID)
	if len(edges) == 0 {
		return ""
	}

	best := edges[0]
	for _, e := range edges[1:] {
		if e.Weight > best.Weight {
			best = e
		}
	}
	return best.To
}
