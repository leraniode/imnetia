// Package types defines the shared types used by both the public engine
// package and the internal runtime.
//
// These types are here — not in imnetia/engine — to break the import cycle
// that would otherwise exist between imnetia/engine (which imports
// internal/runtime) and internal/runtime (which needs NodeHandler, NodeResult,
// and Context).
//
// Developers who write NodeHandlers import this package directly:
//
//	import "github.com/leraniode/imnetia/types"
//
//	var _ types.NodeHandler = myHandler // type-check at compile time
package types

import (
	"fmt"

	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
	"github.com/leraniode/x/centrix/registry"
)

// ─── NodeResult ───────────────────────────────────────────────────────────────

// NodeResult is what a NodeHandler returns alongside the transformed Signal.
// It carries only routing — which node should run next.
//
// The Signal IS the output. NodeResult is purely a routing hint for the executor.
type NodeResult struct {
	// Next is the ID of the node to execute after this one.
	// Empty string means "follow the highest-weight outgoing edge."
	Next string
}

// ─── NodeHandler ──────────────────────────────────────────────────────────────

// NodeHandler is the function a developer writes for each Node.
//
// It receives the run-scoped Context and the current Signal,
// and returns a transformed Signal and a routing hint.
//
// NodeHandlers must be deterministic — same inputs always produce
// the same outputs. They must not retain references to the Signal
// after returning.
type NodeHandler func(ctx Context, in core.Signal) (core.Signal, NodeResult)

// ─── Context ──────────────────────────────────────────────────────────────────

// Context is the run-scoped operational store.
//
// It carries developer-supplied data that is not part of the feature
// space — session IDs, raw input strings, configuration flags, metadata.
// It also provides scoped access to the KnowledgeStore via Knowledge()
// and the Registry via FeatureName() for debugging and trace readability.
//
// Signal carries the mathematical reasoning state (Vector, Confidence,
// Trace). Context carries everything else.
//
// Context is not safe for concurrent use across goroutines. In the
// pipeline execution model each node receives Context sequentially.
type Context struct {
	values map[string]any
	store  *knowledge.KnowledgeStore
	reg    *registry.Registry
}

// NewContext creates a Context backed by the given KnowledgeStore and Registry.
// Pass nil for reg if Registry lookups are not needed in NodeHandlers.
func NewContext(store *knowledge.KnowledgeStore, reg *registry.Registry) Context {
	return Context{
		values: make(map[string]any),
		store:  store,
		reg:    reg,
	}
}

// Set stores a value under key.
func (c *Context) Set(key string, value any) {
	c.values[key] = value
}

// Get returns the value for key. Returns nil, false if absent.
func (c Context) Get(key string) (any, bool) {
	v, ok := c.values[key]
	return v, ok
}

// Has reports whether a value exists for key.
func (c Context) Has(key string) bool {
	_, ok := c.values[key]
	return ok
}

// String returns the value for key as a string, or "" if absent/wrong type.
func (c Context) String(key string) string {
	v, ok := c.values[key]
	if !ok {
		return ""
	}
	s, _ := v.(string)
	return s
}

// Int returns the value for key as an int, or 0 if absent/wrong type.
func (c Context) Int(key string) int {
	v, ok := c.values[key]
	if !ok {
		return 0
	}
	n, _ := v.(int)
	return n
}

// Float returns the value for key as a float64, or 0 if absent/wrong type.
func (c Context) Float(key string) float64 {
	v, ok := c.values[key]
	if !ok {
		return 0
	}
	f, _ := v.(float64)
	return f
}

// Bool returns the value for key as a bool, or false if absent/wrong type.
func (c Context) Bool(key string) bool {
	v, ok := c.values[key]
	if !ok {
		return false
	}
	b, _ := v.(bool)
	return b
}

// Vector returns the value for key as a core.SparseVector, or nil.
//
// The reserved key "input" is read by the ContextEngine to construct
// the initial Signal. Set it before calling Engine.Run:
//
//	ctx.Set("input", core.SparseVector{1: 0.9, 2: 0.7})
func (c Context) Vector(key string) core.SparseVector {
	v, ok := c.values[key]
	if !ok {
		return nil
	}
	sv, _ := v.(core.SparseVector)
	return sv
}

// Knowledge returns all Prototype entries for the given domain,
// sorted by weight descending (highest-trust first).
//
// Nodes use the returned entries with Centrix algebra (Cosine, Dot,
// Generate) to find what is relevant to their current Signal.
func (c Context) Knowledge(domain string) ([]knowledge.Entry, error) {
	if c.store == nil {
		return nil, fmt.Errorf("imnetia/types: Context has no KnowledgeStore")
	}
	return c.store.Domain(domain)
}

// Store returns the underlying KnowledgeStore.
// Used internally by the runtime. Prefer Knowledge() in NodeHandlers.
func (c Context) Store() *knowledge.KnowledgeStore {
	return c.store
}

// FeatureName returns the human-readable name for a FeatureIndex.
// Uses the Registry to reverse-lookup concept names.
// Returns "?" if the index is not registered or no Registry is set.
// Use this in NodeHandlers for debugging and trace readability.
func (c Context) FeatureName(idx core.FeatureIndex) string {
	if c.reg == nil {
		return "?"
	}
	name, ok := c.reg.Name(idx)
	if !ok {
		return "?"
	}
	return name
}

// Reg returns the underlying Registry.
// Used for direct Registry access when FeatureName is not sufficient.
// Returns nil if no Registry was provided.
func (c Context) Reg() *registry.Registry {
	return c.reg
}
