package graph

import (
	"testing"
)

func TestNewNode(t *testing.T) {
	n, err := NewNode("classify", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n.ID != "classify" {
		t.Fatalf("expected ID %q, got %q", "classify", n.ID)
	}
}

func TestNewNodeEmptyID(t *testing.T) {
	_, err := NewNode("", nil)
	if err == nil {
		t.Fatal("expected error for empty ID")
	}
}

func TestFlowAddAndLink(t *testing.T) {
	f := NewFlow()

	a, _ := NewNode("a", nil)
	b, _ := NewNode("b", nil)

	if err := f.Add(a); err != nil {
		t.Fatalf("Add a: %v", err)
	}
	if err := f.Add(b); err != nil {
		t.Fatalf("Add b: %v", err)
	}
	if err := f.Entry("a"); err != nil {
		t.Fatalf("Entry: %v", err)
	}
	if err := f.Link("a", "b", 0.8, 1.0); err != nil {
		t.Fatalf("Link: %v", err)
	}

	edges := f.OutEdges("a")
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].To != "b" {
		t.Fatalf("expected edge to %q, got %q", "b", edges[0].To)
	}
	if edges[0].Weight != 0.8 {
		t.Fatalf("expected weight 0.8, got %v", edges[0].Weight)
	}
}

func TestFlowDuplicateNode(t *testing.T) {
	f := NewFlow()
	a, _ := NewNode("a", nil)
	f.Add(a)
	err := f.Add(a)
	if err == nil {
		t.Fatal("expected duplicate node error")
	}
}

func TestFlowSelfLoop(t *testing.T) {
	f := NewFlow()
	a, _ := NewNode("a", nil)
	f.Add(a)
	err := f.Link("a", "a", 1.0, 1.0)
	if err == nil {
		t.Fatal("expected self-loop error")
	}
}

func TestFlowValidate(t *testing.T) {
	f := NewFlow()
	if err := f.Validate(); err != ErrEmptyFlow {
		t.Fatalf("expected ErrEmptyFlow, got %v", err)
	}

	a, _ := NewNode("a", nil)
	f.Add(a)
	if err := f.Validate(); err != ErrNoEntry {
		t.Fatalf("expected ErrNoEntry, got %v", err)
	}

	f.Entry("a")
	if err := f.Validate(); err != nil {
		t.Fatalf("expected valid flow, got %v", err)
	}
}

func TestEdgeWeightClamped(t *testing.T) {
	f := NewFlow()
	a, _ := NewNode("a", nil)
	b, _ := NewNode("b", nil)
	f.Add(a)
	f.Add(b)
	f.Link("a", "b", 1.5, -0.5) // out of range

	edges := f.OutEdges("a")
	if edges[0].Weight != 1.0 {
		t.Fatalf("expected weight clamped to 1.0, got %v", edges[0].Weight)
	}
	if edges[0].Trust != 0.0 {
		t.Fatalf("expected trust clamped to 0.0, got %v", edges[0].Trust)
	}
}

func TestUpdateEdge(t *testing.T) {
	f := NewFlow()
	a, _ := NewNode("a", nil)
	b, _ := NewNode("b", nil)
	f.Add(a)
	f.Add(b)
	f.Link("a", "b", 0.5, 0.5)

	if err := f.UpdateEdge("a", "b", 0.9, 0.8); err != nil {
		t.Fatalf("UpdateEdge: %v", err)
	}
	edges := f.OutEdges("a")
	if edges[0].Weight != 0.9 {
		t.Fatalf("expected weight 0.9, got %v", edges[0].Weight)
	}
	if edges[0].Trust != 0.8 {
		t.Fatalf("expected trust 0.8, got %v", edges[0].Trust)
	}
}
