package learning

import (
	"os"
	"testing"

	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

// buildFlow builds a simple a→b flow with weight=0.5, trust=0.5.
func buildFlow() *graph.Flow {
	f := graph.NewFlow()
	a, _ := graph.NewNode("a", nil)
	b, _ := graph.NewNode("b", nil)
	f.Add(a)
	f.Add(b)
	f.Link("a", "b", 0.5, 0.5)
	f.Entry("a")
	return f
}

// buildStore builds a KnowledgeStore with one Intelligence domain.
func buildStore() *knowledge.KnowledgeStore {
	ks := knowledge.New()
	ks.Load("intent", knowledge.Intelligence, []knowledge.Entry{
		{
			ID:        "intent.greet",
			Prototype: core.NewPrototype(core.SparseVector{1: 1.0}, 0.5),
		},
	})
	return ks
}

// buildSignal builds a Signal with a two-step Trace simulating a→b execution.
func buildSignal() core.Signal {
	s := core.NewSignalFromVector(core.SparseVector{1: 0.9}, 0.8)
	s = core.Attenuate(s, 0.05, "a")
	s = core.Attenuate(s, 0.05, "b")
	return s
}

func TestUpdateStrengthensOnCorrect(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	u := New(f, ks)
	sig := buildSignal()

	before := f.OutEdges("a")[0].Weight

	if err := u.Update(sig, 1.0); err != nil {
		t.Fatalf("Update: %v", err)
	}

	after := f.OutEdges("a")[0].Weight
	if after <= before {
		t.Fatalf("expected weight to increase on correct outcome: before=%v after=%v", before, after)
	}
}

func TestUpdateWeakensOnIncorrect(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	u := New(f, ks)
	sig := buildSignal()

	before := f.OutEdges("a")[0].Weight

	if err := u.Update(sig, 0.0); err != nil {
		t.Fatalf("Update: %v", err)
	}

	after := f.OutEdges("a")[0].Weight
	if after >= before {
		t.Fatalf("expected weight to decrease on incorrect outcome: before=%v after=%v", before, after)
	}
}

func TestUpdateTrustUpdated(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	u := New(f, ks)
	sig := buildSignal()

	before := f.OutEdges("a")[0].Trust
	u.Update(sig, 1.0)
	after := f.OutEdges("a")[0].Trust

	if after <= before {
		t.Fatalf("expected trust to increase: before=%v after=%v", before, after)
	}
}

func TestUpdateEmptyTrace(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	u := New(f, ks)

	empty := core.NewSignal(0)
	err := u.Update(empty, 1.0)
	if err == nil {
		t.Fatal("expected ErrEmptyTrace")
	}
}

func TestUpdateInvalidOutcome(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	u := New(f, ks)
	sig := buildSignal()

	if err := u.Update(sig, 1.5); err == nil {
		t.Fatal("expected ErrInvalidOutcome for outcome > 1")
	}
	if err := u.Update(sig, -0.1); err == nil {
		t.Fatal("expected ErrInvalidOutcome for outcome < 0")
	}
}

func TestSnapshotRoundTrip(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	u := New(f, ks)
	sig := buildSignal()

	// Apply a learning update so state changes.
	u.Update(sig, 1.0)
	weightAfterUpdate := f.OutEdges("a")[0].Weight

	// Take snapshot.
	snap, err := Take(f, ks)
	if err != nil {
		t.Fatalf("Take: %v", err)
	}
	if len(snap.Edges) == 0 {
		t.Fatal("expected edges in snapshot")
	}

	// Apply another update to change state again.
	u.Update(sig, 0.0)
	weightAfterSecond := f.OutEdges("a")[0].Weight
	if weightAfterSecond >= weightAfterUpdate {
		t.Fatal("expected weight to decrease after incorrect outcome")
	}

	// Restore from snapshot — should go back to weightAfterUpdate.
	if err := Restore(snap, f, ks); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	restored := f.OutEdges("a")[0].Weight
	if restored != weightAfterUpdate {
		t.Fatalf("expected restored weight %v, got %v", weightAfterUpdate, restored)
	}
}

func TestSnapshotSaveLoad(t *testing.T) {
	f := buildFlow()
	ks := buildStore()

	snap, _ := Take(f, ks)

	tmp, err := os.CreateTemp("", "snap-*.json")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	tmp.Close()
	defer os.Remove(tmp.Name())

	if err := Save(snap, tmp.Name()); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(tmp.Name())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Version != snap.Version {
		t.Fatalf("version mismatch: %v vs %v", loaded.Version, snap.Version)
	}
	if len(loaded.Edges) != len(snap.Edges) {
		t.Fatalf("edge count mismatch")
	}
}

func TestSnapshotExcludesFoundation(t *testing.T) {
	f := buildFlow()
	ks := knowledge.New()
	ks.Load("facts", knowledge.Foundation, []knowledge.Entry{
		{ID: "fact.one", Prototype: core.NewPrototype(core.SparseVector{1: 1.0}, 1.0)},
	})
	ks.Load("learned", knowledge.Intelligence, []knowledge.Entry{
		{ID: "pattern.a", Prototype: core.NewPrototype(core.SparseVector{2: 1.0}, 0.5)},
	})

	snap, _ := Take(f, ks)

	for _, p := range snap.Prototypes {
		if p.Domain == "facts" {
			t.Fatal("Foundation domain should not appear in snapshot")
		}
	}
}

func TestWithRatesClamps(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	u := WithRates(f, ks, -1.0, 2.0) // both out of range
	if u.alpha <= 0 || u.alpha >= 1 {
		t.Fatalf("alpha not clamped: %v", u.alpha)
	}
	if u.eta <= 0 || u.eta >= 1 {
		t.Fatalf("eta not clamped: %v", u.eta)
	}
}
