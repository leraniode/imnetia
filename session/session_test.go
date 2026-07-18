package session

import (
	"testing"

	"github.com/leraniode/x/centrix/core"
)

func makeSignal(features map[uint32]float64, confidence float64) core.Signal {
	return core.NewSignalFromVector(core.SparseVector(features), confidence)
}

func TestRecordSingleTurn(t *testing.T) {
	s := New(DefaultDecay)
	sig := makeSignal(map[uint32]float64{1: 0.9, 2: 0.8}, 0.85)

	if err := s.Record(sig, 0); err != nil {
		t.Fatalf("Record: %v", err)
	}
	if s.Len() != 1 {
		t.Fatalf("expected 1 turn, got %d", s.Len())
	}
}

func TestRecordMultipleTurns(t *testing.T) {
	s := New(DefaultDecay)

	s.Record(makeSignal(map[uint32]float64{1: 0.9}, 0.8), 0)
	s.Record(makeSignal(map[uint32]float64{2: 0.9}, 0.85), 1)
	s.Record(makeSignal(map[uint32]float64{3: 0.9}, 0.9), 2)

	if s.Len() != 3 {
		t.Fatalf("expected 3 turns, got %d", s.Len())
	}
}

func TestDecayApplied(t *testing.T) {
	s := New(DefaultDecay)

	// Record turn 0
	s.Record(makeSignal(map[uint32]float64{1: 0.9}, 0.8), 0)

	// Get weight of turn 0 before turn 1
	entries, _ := s.ks.Domain(SessionDomain)
	if len(entries) == 0 {
		t.Fatal("expected entries in session domain")
	}
	weightBefore := entries[0].Prototype.Weight

	// Record turn 1 — should decay turn 0
	s.Record(makeSignal(map[uint32]float64{2: 0.9}, 0.85), 1)

	entries, _ = s.ks.Domain(SessionDomain)
	var turn0Weight float64
	for _, e := range entries {
		if e.ID == "turn.0" {
			turn0Weight = e.Prototype.Weight
		}
	}

	if turn0Weight >= weightBefore {
		t.Fatalf("expected turn 0 weight to decay: before=%v after=%v",
			weightBefore, turn0Weight)
	}
}

func TestMostRecent(t *testing.T) {
	s := New(DefaultDecay)

	_, ok := s.MostRecent()
	if ok {
		t.Fatal("expected false for empty store")
	}

	sig1 := makeSignal(map[uint32]float64{1: 0.9}, 0.8)
	sig2 := makeSignal(map[uint32]float64{2: 0.9}, 0.9)

	s.Record(sig1, 0)
	s.Record(sig2, 1)

	recent, ok := s.MostRecent()
	if !ok {
		t.Fatal("expected true for non-empty store")
	}
	if recent.Confidence != sig2.Confidence {
		t.Fatalf("expected most recent confidence %v, got %v",
			sig2.Confidence, recent.Confidence)
	}
}

func TestSessionQueryable(t *testing.T) {
	s := New(DefaultDecay)
	s.Record(makeSignal(map[uint32]float64{1: 0.9, 2: 0.7}, 0.85), 0)

	// Session store should be queryable via KnowledgeStore
	entries, err := s.KnowledgeStore().Domain(SessionDomain)
	if err != nil {
		t.Fatalf("Domain: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected entries queryable from session store")
	}
}

func TestClear(t *testing.T) {
	s := New(DefaultDecay)
	s.Record(makeSignal(map[uint32]float64{1: 0.9}, 0.8), 0)
	s.Record(makeSignal(map[uint32]float64{2: 0.9}, 0.85), 1)

	s.Clear()

	if s.Len() != 0 {
		t.Fatalf("expected 0 turns after Clear, got %d", s.Len())
	}

	// Domain should be gone after clear
	_, err := s.KnowledgeStore().Domain(SessionDomain)
	if err == nil {
		t.Fatal("expected domain not found after Clear")
	}
}

func TestTurnsReturned(t *testing.T) {
	s := New(DefaultDecay)
	s.Record(makeSignal(map[uint32]float64{1: 0.9}, 0.8), 0)
	s.Record(makeSignal(map[uint32]float64{2: 0.9}, 0.85), 1)

	turns := s.Turns()
	if len(turns) != 2 {
		t.Fatalf("expected 2 turns, got %d", len(turns))
	}
	if turns[0].Index != 0 || turns[1].Index != 1 {
		t.Fatal("turns not in order")
	}
}

func TestEmptySignalNotRecorded(t *testing.T) {
	s := New(DefaultDecay)
	empty := core.NewSignal(0)
	s.Record(empty, 0)

	if s.Len() != 0 {
		t.Fatal("empty signal should not be recorded")
	}
}

func TestDecayRateClamped(t *testing.T) {
	// Out-of-range decay should use DefaultDecay
	s1 := New(-0.5)
	s2 := New(1.5)
	if s1.decay != DefaultDecay {
		t.Fatalf("expected decay clamped to default, got %v", s1.decay)
	}
	if s2.decay != DefaultDecay {
		t.Fatalf("expected decay clamped to default, got %v", s2.decay)
	}
}
