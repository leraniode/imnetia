// Package graph defines the structure of an Imnetia reasoning pipeline.
//
// A Flow is a directed weighted graph of Nodes. It defines structure only —
// which nodes exist, how they connect, and the weight and trust on each
// connection. No execution logic lives here.
//
// The developer builds a Flow, adds Nodes, links them, and hands it to the
// runtime. The graph package has no knowledge of Signals, execution, or
// knowledge stores.
package graph

import (
	"errors"
	"fmt"
)

// ─── Errors ───────────────────────────────────────────────────────────────────

var (
	// ErrNodeNotFound is returned when a referenced node ID does not exist in the Flow.
	ErrNodeNotFound = errors.New("imnetia/graph: node not found")

	// ErrDuplicateNode is returned when a node with the same ID is added twice.
	ErrDuplicateNode = errors.New("imnetia/graph: duplicate node ID")

	// ErrNoEntry is returned when Run is called on a Flow with no entry point set.
	ErrNoEntry = errors.New("imnetia/graph: no entry node set")

	// ErrEmptyFlow is returned when Run is called on a Flow with no nodes.
	ErrEmptyFlow = errors.New("imnetia/graph: flow has no nodes")

	// ErrSelfLoop is returned when a node is linked to itself.
	ErrSelfLoop = errors.New("imnetia/graph: self-loop not allowed")
)

// ─── Node ─────────────────────────────────────────────────────────────────────

// Node is the atomic unit of reasoning in a Flow.
// It holds an ID and a Handler — the function the runtime calls during execution.
// The Handler is defined by the developer; its signature is declared in the
// runtime package, which owns execution semantics.
//
// Node carries no execution state. It is a structural definition.
type Node struct {
	ID      string
	Handler any // runtime.NodeHandler — typed in the runtime package to avoid circular imports
}

// NewNode creates a Node with the given ID and handler.
// The handler type is validated by the runtime, not here.
// ID must not be empty.
func NewNode(id string, handler any) (*Node, error) {
	if id == "" {
		return nil, fmt.Errorf("imnetia/graph: node ID must not be empty")
	}
	return &Node{ID: id, Handler: handler}, nil
}

// ─── Edge ─────────────────────────────────────────────────────────────────────

// Edge is a directed connection between two Nodes.
//
// Weight and Trust are independent dimensions:
//   - Weight is routing strength — how confidently should the runtime follow this edge?
//   - Trust is reliability — how much has this connection proven correct over time?
//
// They evolve independently through the learning system.
// A strong connection (high Weight) can be unreliable (low Trust).
// A weak connection (low Weight) can be highly trusted.
type Edge struct {
	From   string
	To     string
	Weight float64 // routing strength, [0.0, 1.0]
	Trust  float64 // reliability, [0.0, 1.0] — updated by learning
}

// ─── Flow ─────────────────────────────────────────────────────────────────────

// Flow is a directed weighted graph of Nodes.
// It defines the reasoning pipeline — which nodes exist and how they connect.
//
// A Flow is not safe for concurrent modification. Build the Flow before
// handing it to the runtime. The runtime reads the Flow but never writes to it.
type Flow struct {
	nodes   map[string]*Node
	edges   map[string][]Edge // from node ID → outgoing edges
	entry   string            // entry node ID
}

// NewFlow creates an empty Flow.
func NewFlow() *Flow {
	return &Flow{
		nodes: make(map[string]*Node),
		edges: make(map[string][]Edge),
	}
}

// Add registers a Node in the Flow.
// Returns ErrDuplicateNode if a node with the same ID already exists.
func (f *Flow) Add(n *Node) error {
	if n == nil {
		return fmt.Errorf("imnetia/graph: node must not be nil")
	}
	if _, exists := f.nodes[n.ID]; exists {
		return fmt.Errorf("%w: %q", ErrDuplicateNode, n.ID)
	}
	f.nodes[n.ID] = n
	return nil
}

// Link creates a directed Edge from node `from` to node `to`.
// Both nodes must already be registered in the Flow.
// Self-loops are not allowed.
// Weight and Trust are clamped to [0.0, 1.0].
func (f *Flow) Link(from, to string, weight, trust float64) error {
	if from == to {
		return fmt.Errorf("%w: %q → %q", ErrSelfLoop, from, to)
	}
	if _, ok := f.nodes[from]; !ok {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, from)
	}
	if _, ok := f.nodes[to]; !ok {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, to)
	}
	e := Edge{
		From:   from,
		To:     to,
		Weight: clamp01(weight),
		Trust:  clamp01(trust),
	}
	f.edges[from] = append(f.edges[from], e)
	return nil
}

// Entry sets the entry node for flow execution.
// The node must already be registered.
func (f *Flow) Entry(id string) error {
	if _, ok := f.nodes[id]; !ok {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, id)
	}
	f.entry = id
	return nil
}

// EntryID returns the ID of the entry node.
func (f *Flow) EntryID() string {
	return f.entry
}

// Node returns the Node with the given ID, or nil and false if not found.
func (f *Flow) Node(id string) (*Node, bool) {
	n, ok := f.nodes[id]
	return n, ok
}

// OutEdges returns the outgoing edges from the given node ID.
// Returns nil if the node has no outgoing edges or does not exist.
func (f *Flow) OutEdges(nodeID string) []Edge {
	return f.edges[nodeID]
}

// Nodes returns all registered node IDs. Order is not guaranteed.
func (f *Flow) Nodes() []string {
	ids := make([]string, 0, len(f.nodes))
	for id := range f.nodes {
		ids = append(ids, id)
	}
	return ids
}

// Len returns the number of nodes in the Flow.
func (f *Flow) Len() int {
	return len(f.nodes)
}

// Validate checks that the Flow is executable:
//   - at least one node
//   - entry node is set and exists
//
// Does not check for cycles — the runtime handles cycle detection during execution.
func (f *Flow) Validate() error {
	if len(f.nodes) == 0 {
		return ErrEmptyFlow
	}
	if f.entry == "" {
		return ErrNoEntry
	}
	if _, ok := f.nodes[f.entry]; !ok {
		return fmt.Errorf("%w: entry %q", ErrNodeNotFound, f.entry)
	}
	return nil
}

// ─── Weight update ────────────────────────────────────────────────────────────

// UpdateEdge updates the Weight and Trust of an existing edge in place.
// Used by the learning system after a run completes.
// Returns ErrNodeNotFound if the from node does not exist.
// Returns an error if the edge from→to does not exist.
func (f *Flow) UpdateEdge(from, to string, weight, trust float64) error {
	edges, ok := f.edges[from]
	if !ok {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, from)
	}
	for i := range edges {
		if edges[i].To == to {
			edges[i].Weight = clamp01(weight)
			edges[i].Trust = clamp01(trust)
			f.edges[from] = edges
			return nil
		}
	}
	return fmt.Errorf("imnetia/graph: edge %q → %q not found", from, to)
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
