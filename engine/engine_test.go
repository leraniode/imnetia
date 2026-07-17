package engine_test

import (
	"testing"

	"github.com/leraniode/imnetia/engine"
	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/imnetia/types"
	"github.com/leraniode/x/centrix/core"
)

// makeStore builds a KnowledgeStore with one Foundation domain.
func makeStore() *knowledge.KnowledgeStore {
	ks := knowledge.New()
	ks.Load("intent", knowledge.Foundation, []knowledge.Entry{
		{
			ID: "intent.greet",
			Prototype: core.NewPrototype(
				core.SparseVector{1: 1.0, 2: 0.8},
				0.9,
			),
		},
		{
			ID: "intent.farewell",
			Prototype: core.NewPrototype(
				core.SparseVector{3: 1.0, 4: 0.7},
				0.8,
			),
		},
	})
	return ks
}

// classifyHandler is a simple NodeHandler that finds the top Prototype
// match in the "intent" domain and stores the matched ID in Context.
func classifyHandler(ctx types.Context, in core.Signal) (core.Signal, types.NodeResult) {
	entries, err := ctx.Knowledge("intent")
	if err != nil || len(entries) == 0 {
		return in, types.NodeResult{}
	}

	// Find best Cosine match manually.
	var bestID string
	var bestSim float64
	for _, e := range entries {
		sim := core.Cosine(in.Vector, e.Prototype.Vector)
		if sim > bestSim {
			bestSim = sim
			bestID = e.ID
		}
	}

	// Attenuate signal slightly to show transformation.
	out := core.Attenuate(in, 0.1, "classify")
	_ = bestID // would normally set result in a real system
	return out, types.NodeResult{}
}

func TestEngineRunSingleNode(t *testing.T) {
	ks := makeStore()

	f := graph.NewFlow()
	n, _ := graph.NewNode("classify", types.NodeHandler(classifyHandler))
	f.Add(n)
	f.Entry("classify")

	e := engine.New(ks)
	ctx := types.NewContext(ks)
	ctx.Set("input", core.SparseVector{1: 0.9, 2: 0.7}) // close to intent.greet

	sig, err := e.Run(f, ctx)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if sig.Confidence <= 0 {
		t.Fatalf("expected positive confidence, got %v", sig.Confidence)
	}
	if len(sig.Trace.Steps()) == 0 {
		t.Fatalf("expected non-empty trace")
	}
}

func TestEngineRunChain(t *testing.T) {
	ks := makeStore()

	f := graph.NewFlow()
	a, _ := graph.NewNode("a", types.NodeHandler(func(ctx types.Context, in core.Signal) (core.Signal, types.NodeResult) {
		out := core.Attenuate(in, 0.05, "a")
		return out, types.NodeResult{Next: "b"}
	}))
	b, _ := graph.NewNode("b", types.NodeHandler(func(ctx types.Context, in core.Signal) (core.Signal, types.NodeResult) {
		out := core.Attenuate(in, 0.05, "b")
		return out, types.NodeResult{}
	}))

	f.Add(a)
	f.Add(b)
	f.Link("a", "b", 1.0, 1.0)
	f.Entry("a")

	e := engine.New(ks)
	ctx := types.NewContext(ks)
	ctx.Set("input", core.SparseVector{1: 1.0})

	sig, err := e.Run(f, ctx)
	if err != nil {
		t.Fatalf("Run chain: %v", err)
	}
	// Two nodes ran — trace should have at least 2 steps
	if len(sig.Trace.Steps()) < 2 {
		t.Fatalf("expected at least 2 trace steps, got %d", len(sig.Trace.Steps()))
	}
}

func TestEngineRunEmptyFlow(t *testing.T) {
	ks := makeStore()
	f := graph.NewFlow()
	e := engine.New(ks)
	ctx := types.NewContext(ks)

	_, err := e.Run(f, ctx)
	if err == nil {
		t.Fatal("expected error for empty flow")
	}
}

func TestEngineRunNoStore(t *testing.T) {
	e := engine.New(nil)
	f := graph.NewFlow()
	ctx := types.NewContext(nil)

	_, err := e.Run(f, ctx)
	if err == nil {
		t.Fatal("expected ErrNoStore")
	}
}

func TestEngineRunCycleDetection(t *testing.T) {
	ks := makeStore()

	f := graph.NewFlow()
	// Node always routes back to itself
	a, _ := graph.NewNode("a", types.NodeHandler(func(ctx types.Context, in core.Signal) (core.Signal, types.NodeResult) {
		return in, types.NodeResult{Next: "a"}
	}))
	f.Add(a)
	f.Entry("a")

	e := engine.New(ks)
	ctx := types.NewContext(ks)
	ctx.Set("input", core.SparseVector{1: 1.0})

	_, err := e.Run(f, ctx)
	if err == nil {
		t.Fatal("expected cycle detection error")
	}
}

func TestContextKnowledge(t *testing.T) {
	ks := makeStore()
	ctx := types.NewContext(ks)

	entries, err := ctx.Knowledge("intent")
	if err != nil {
		t.Fatalf("Knowledge: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
}

func TestContextTypedAccessors(t *testing.T) {
	ks := makeStore()
	ctx := types.NewContext(ks)

	ctx.Set("name", "imnetia")
	ctx.Set("count", 42)
	ctx.Set("score", 0.95)
	ctx.Set("active", true)

	if ctx.String("name") != "imnetia" {
		t.Fatalf("String accessor failed")
	}
	if ctx.Int("count") != 42 {
		t.Fatalf("Int accessor failed")
	}
	if ctx.Float("score") != 0.95 {
		t.Fatalf("Float accessor failed")
	}
	if !ctx.Bool("active") {
		t.Fatalf("Bool accessor failed")
	}
	if ctx.String("missing") != "" {
		t.Fatalf("missing key should return zero value")
	}
}
