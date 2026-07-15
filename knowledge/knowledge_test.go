package knowledge

import (
	"testing"

	"github.com/leraniode/xgo/centrix/core"
)

func makeEntry(id string, weight float64, features map[uint32]float64) Entry {
	sv := core.SparseVector(features)
	return Entry{
		ID:        id,
		Prototype: core.NewPrototype(sv, weight),
	}
}

func TestLoadAndDomain(t *testing.T) {
	ks := New()

	entries := []Entry{
		makeEntry("intent.greet", 0.9, map[uint32]float64{1: 1.0, 2: 0.8}),
		makeEntry("intent.farewell", 0.7, map[uint32]float64{3: 1.0, 4: 0.6}),
	}

	if err := ks.Load("intent", Foundation, entries); err != nil {
		t.Fatalf("Load: %v", err)
	}

	got, err := ks.Domain("intent")
	if err != nil {
		t.Fatalf("Domain: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(got))
	}
	// Should be sorted by weight descending — greet (0.9) first
	if got[0].ID != "intent.greet" {
		t.Fatalf("expected greet first, got %q", got[0].ID)
	}
}

func TestFoundationSealed(t *testing.T) {
	ks := New()
	ks.Load("intent", Foundation, []Entry{
		makeEntry("intent.greet", 0.9, map[uint32]float64{1: 1.0}),
	})

	// Try adding to sealed Foundation domain
	err := ks.Load("intent", Foundation, []Entry{
		makeEntry("intent.new", 0.5, map[uint32]float64{5: 1.0}),
	})
	if err == nil {
		t.Fatal("expected sealed error for Foundation domain")
	}
}

func TestIntelligenceDomainUpdatable(t *testing.T) {
	ks := New()
	ks.Load("learned", Intelligence, []Entry{
		makeEntry("pattern.a", 0.5, map[uint32]float64{10: 1.0}),
	})

	// Intelligence domains can receive new entries
	err := ks.Load("learned", Intelligence, []Entry{
		makeEntry("pattern.b", 0.6, map[uint32]float64{11: 1.0}),
	})
	if err != nil {
		t.Fatalf("unexpected error adding to Intelligence domain: %v", err)
	}

	got, _ := ks.Domain("learned")
	if len(got) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(got))
	}
}

func TestUpdateWeight(t *testing.T) {
	ks := New()
	ks.Load("learned", Intelligence, []Entry{
		makeEntry("pattern.a", 0.5, map[uint32]float64{10: 1.0}),
	})

	if err := ks.UpdateWeight("learned", "pattern.a", 0.9); err != nil {
		t.Fatalf("UpdateWeight: %v", err)
	}

	e, err := ks.Get("learned", "pattern.a")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if e.Prototype.Weight != 0.9 {
		t.Fatalf("expected weight 0.9, got %v", e.Prototype.Weight)
	}
}

func TestUpdateWeightFoundationRejected(t *testing.T) {
	ks := New()
	ks.Load("facts", Foundation, []Entry{
		makeEntry("fact.one", 1.0, map[uint32]float64{1: 1.0}),
	})

	err := ks.UpdateWeight("facts", "fact.one", 0.5)
	if err == nil {
		t.Fatal("expected error updating Foundation domain weight")
	}
}

func TestDomainNotFound(t *testing.T) {
	ks := New()
	_, err := ks.Domain("nonexistent")
	if err == nil {
		t.Fatal("expected ErrDomainNotFound")
	}
}

func TestGetNotFound(t *testing.T) {
	ks := New()
	ks.Load("intent", Foundation, []Entry{
		makeEntry("intent.greet", 0.9, map[uint32]float64{1: 1.0}),
	})

	_, err := ks.Get("intent", "intent.missing")
	if err == nil {
		t.Fatal("expected ErrPrototypeNotFound")
	}
}

func TestLazyLoad(t *testing.T) {
	ks := NewLazy()
	loaded := false

	ks.Register("intent", Foundation, func() ([]Entry, error) {
		loaded = true
		return []Entry{
			makeEntry("intent.greet", 0.9, map[uint32]float64{1: 1.0}),
		}, nil
	})

	if loaded {
		t.Fatal("loader should not have been called before Domain()")
	}

	entries, err := ks.Domain("intent")
	if err != nil {
		t.Fatalf("Domain: %v", err)
	}
	if !loaded {
		t.Fatal("loader should have been called on first Domain()")
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}

	// Second call should not trigger loader again
	loaded = false
	ks.Domain("intent")
	if loaded {
		t.Fatal("loader should not be called again after first load")
	}
}

func TestSize(t *testing.T) {
	ks := New()
	ks.Load("a", Foundation, []Entry{
		makeEntry("a.1", 0.9, map[uint32]float64{1: 1.0}),
		makeEntry("a.2", 0.8, map[uint32]float64{2: 1.0}),
	})
	ks.Load("b", Intelligence, []Entry{
		makeEntry("b.1", 0.7, map[uint32]float64{3: 1.0}),
	})

	if ks.Size() != 3 {
		t.Fatalf("expected size 3, got %d", ks.Size())
	}
}

func TestSnapshot(t *testing.T) {
	ks := New()
	ks.Load("intent", Intelligence, []Entry{
		makeEntry("intent.greet", 0.9, map[uint32]float64{1: 1.0}),
	})

	entries, kind, err := ks.Snapshot("intent")
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if kind != Intelligence {
		t.Fatalf("expected Intelligence kind")
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry in snapshot")
	}
}
