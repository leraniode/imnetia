// Package learning applies bounded weight updates to a Flow and KnowledgeStore
// after a run completes, based on the outcome and the Signal's Trace.
//
// Learning reads the Trace — the complete execution log carried by the Signal —
// and applies the delta rule to every edge and Prototype that participated
// in that run. This is how a system improves over time: each run produces
// a Trace, each Trace drives a learning update, each update is bounded,
// auditable, and reversible via snapshot.
//
// Authority: learning operates on Intelligence knowledge only.
// Foundation domains are immutable — learning never touches them.
//
// The learning package does not run the flow. It only reads the outcome
// of a run that already happened. The developer calls Update after Engine.Run.
package learning

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

// ─── Constants ────────────────────────────────────────────────────────────────

const (
	// DefaultAlpha is the learning rate for the delta rule.
	// Conservative — weights move slowly toward the target.
	DefaultAlpha = 0.10

	// DefaultTrustEta is the learning rate for Trust updates.
	// Slightly faster than weight updates — Trust should respond
	// to reliability signals quickly.
	DefaultTrustEta = 0.15
)

// ─── Errors ───────────────────────────────────────────────────────────────────

var (
	ErrEmptyTrace     = errors.New("imnetia/learning: signal has empty trace — nothing to learn from")
	ErrInvalidOutcome = errors.New("imnetia/learning: outcome must be in [0.0, 1.0]")
)

// ─── Updater ──────────────────────────────────────────────────────────────────

// Updater applies learning updates after a run.
// It holds the learning rates and references to the Flow and KnowledgeStore
// that should be updated.
type Updater struct {
	flow  *graph.Flow
	store *knowledge.KnowledgeStore
	alpha float64 // delta rule learning rate for edge weights
	eta   float64 // learning rate for Trust updates
}

// New creates an Updater with default learning rates.
func New(flow *graph.Flow, store *knowledge.KnowledgeStore) *Updater {
	return &Updater{
		flow:  flow,
		store: store,
		alpha: DefaultAlpha,
		eta:   DefaultTrustEta,
	}
}

// WithRates creates an Updater with custom learning rates.
// Both rates are clamped to (0.0, 1.0).
func WithRates(flow *graph.Flow, store *knowledge.KnowledgeStore, alpha, eta float64) *Updater {
	return &Updater{
		flow:  flow,
		store: store,
		alpha: clampRate(alpha),
		eta:   clampRate(eta),
	}
}

// Update applies bounded learning updates based on the Signal's Trace
// and the given outcome.
//
// outcome is a value in [0.0, 1.0]:
//   - 1.0 = fully correct — strengthen edges and Prototypes that contributed
//   - 0.0 = fully incorrect — weaken edges and Prototypes that contributed
//   - values between model partial correctness (graded feedback)
//
// For each Step in the Trace:
//   - If the step moved from one node to the next, the edge between them
//     receives a Weight and Trust update via the delta rule.
//   - If the step action was Matched or Generated, the Prototype's weight
//     in the Intelligence domain (if present) receives a delta rule update.
//
// Foundation domain Prototypes are never updated — they are immutable.
// Edges between nodes not found in the Flow are silently skipped.
func (u *Updater) Update(signal core.Signal, outcome float64) error {
	if outcome < 0 || outcome > 1 {
		return fmt.Errorf("%w: got %v", ErrInvalidOutcome, outcome)
	}
	if len(signal.Trace.Steps()) == 0 {
		return ErrEmptyTrace
	}

	trace := signal.Trace.Steps()

	// Walk consecutive step pairs to identify which edges were traversed.
	for i := 0; i < len(trace)-1; i++ {
		from := trace[i].Node
		to := trace[i+1].Node

		if from == "" || to == "" || from == to {
			continue
		}

		// Fetch current edge to get Weight and Trust.
		edges := u.flow.OutEdges(from)
		for _, e := range edges {
			if e.To != to {
				continue
			}
			// Delta rule for Weight: ΔW = α × (outcome − W)
			newWeight := clamp01(e.Weight + u.alpha*(outcome-e.Weight))
			// Exponential moving average for Trust.
			newTrust := clamp01(e.Trust + u.eta*(outcome-e.Trust))
			// Write back — UpdateEdge handles clamping too.
			_ = u.flow.UpdateEdge(from, to, newWeight, newTrust)
			break
		}
	}

	// Update Prototype weights for Matched/Generated steps.
	for _, step := range trace {
		if step.Action != core.Matched && step.Action != core.Generated {
			continue
		}
		if step.Value == nil {
			continue
		}
		// step.Value carries the Prototype ID and domain as "domain/id".
		ref, ok := step.Value.(string)
		if !ok || ref == "" {
			continue
		}
		domain, id := splitRef(ref)
		if domain == "" || id == "" {
			continue
		}
		e, err := u.store.Get(domain, id)
		if err != nil {
			// Not found or Foundation — skip silently.
			continue
		}
		newWeight := clamp01(e.Prototype.Weight + u.alpha*(outcome-e.Prototype.Weight))
		_ = u.store.UpdateWeight(domain, id, newWeight)
	}

	return nil
}

// ─── Snapshot ─────────────────────────────────────────────────────────────────

// Snapshot is a point-in-time copy of all learnable state —
// edge weights, Trust values, and Intelligence Prototype weights.
//
// Foundation Prototype weights are not included — they are immutable
// and always loaded from .pack files at startup.
type Snapshot struct {
	Version    int          `json:"version"`
	Timestamp  time.Time    `json:"timestamp"`
	Edges      []edgeState  `json:"edges"`
	Prototypes []protoState `json:"prototypes"`
}

type edgeState struct {
	From   string  `json:"from"`
	To     string  `json:"to"`
	Weight float64 `json:"weight"`
	Trust  float64 `json:"trust"`
}

type protoState struct {
	Domain string  `json:"domain"`
	ID     string  `json:"id"`
	Weight float64 `json:"weight"`
}

// Take captures the current learnable state into a Snapshot.
// Call this after training to preserve progress.
func Take(flow *graph.Flow, store *knowledge.KnowledgeStore) (*Snapshot, error) {
	snap := &Snapshot{
		Version:   1,
		Timestamp: time.Now().UTC(),
	}

	// Capture all edge weights and Trust values.
	for _, nodeID := range flow.Nodes() {
		for _, e := range flow.OutEdges(nodeID) {
			snap.Edges = append(snap.Edges, edgeState{
				From:   e.From,
				To:     e.To,
				Weight: e.Weight,
				Trust:  e.Trust,
			})
		}
	}

	// Capture Intelligence domain Prototype weights.
	for _, domain := range store.Domains() {
		entries, kind, err := store.Snapshot(domain)
		if err != nil {
			continue
		}
		if kind != knowledge.Intelligence {
			continue // Foundation domains are not part of learned state
		}
		for _, e := range entries {
			snap.Prototypes = append(snap.Prototypes, protoState{
				Domain: domain,
				ID:     e.ID,
				Weight: e.Prototype.Weight,
			})
		}
	}

	return snap, nil
}

// Save writes a Snapshot to a file at path as JSON.
func Save(snap *Snapshot, path string) error {
	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return fmt.Errorf("imnetia/learning: marshal snapshot: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("imnetia/learning: write snapshot: %w", err)
	}
	return nil
}

// Load reads a Snapshot from a JSON file at path.
func Load(path string) (*Snapshot, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("imnetia/learning: read snapshot: %w", err)
	}
	var snap Snapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, fmt.Errorf("imnetia/learning: unmarshal snapshot: %w", err)
	}
	return &snap, nil
}

// Restore applies a Snapshot's state back to a Flow and KnowledgeStore.
// This is the rollback mechanism — after a bad run sequence, restore
// the last known-good snapshot.
//
// Edge entries not found in the current Flow are skipped silently.
// Prototype entries for non-existent or Foundation domains are skipped.
func Restore(snap *Snapshot, flow *graph.Flow, store *knowledge.KnowledgeStore) error {
	if snap == nil {
		return errors.New("imnetia/learning: cannot restore nil snapshot")
	}

	for _, e := range snap.Edges {
		_ = flow.UpdateEdge(e.From, e.To, e.Weight, e.Trust)
	}

	for _, p := range snap.Prototypes {
		_ = store.UpdateWeight(p.Domain, p.ID, p.Weight)
	}

	return nil
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func clampRate(v float64) float64 {
	if v <= 0 {
		return 0.01
	}
	if v >= 1 {
		return 0.99
	}
	return v
}

// splitRef splits a "domain/id" reference string.
func splitRef(ref string) (domain, id string) {
	for i, c := range ref {
		if c == '/' {
			return ref[:i], ref[i+1:]
		}
	}
	return "", ""
}
