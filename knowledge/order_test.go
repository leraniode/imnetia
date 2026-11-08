package knowledge

import (
	"reflect"
	"testing"

	"github.com/leraniode/x/centrix/core"
)

// Equal weights must not leave entry order to the sort algorithm: learning
// moves weights between runs, and tied entries should never swap for no reason.
func TestDomain_TiedWeightsOrderedByID(t *testing.T) {
	var entries []Entry
	// Insert in reverse ID order, all with the same weight.
	for _, id := range []string{"e", "d", "c", "b", "a"} {
		entries = append(entries, Entry{
			ID:        id,
			Prototype: core.NewPrototype(core.SparseVector{1: 1}, 0.5),
		})
	}
	ks := New()
	if err := ks.Load("d", Intelligence, entries); err != nil {
		t.Fatal(err)
	}

	ids := func() []string {
		got, err := ks.Domain("d")
		if err != nil {
			t.Fatal(err)
		}
		out := make([]string, len(got))
		for i, e := range got {
			out[i] = e.ID
		}
		return out
	}
	want := []string{"a", "b", "c", "d", "e"}
	for i := 0; i < 50; i++ {
		if got := ids(); !reflect.DeepEqual(got, want) {
			t.Fatalf("tied weights must order by ID: got %v want %v", got, want)
		}
	}
}

func TestDomain_WeightStillDominatesID(t *testing.T) {
	ks := New()
	_ = ks.Load("d", Intelligence, []Entry{
		{ID: "a", Prototype: core.NewPrototype(core.SparseVector{1: 1}, 0.2)},
		{ID: "z", Prototype: core.NewPrototype(core.SparseVector{1: 1}, 0.9)},
	})
	got, _ := ks.Domain("d")
	if got[0].ID != "z" {
		t.Errorf("higher weight must come first, got %s first", got[0].ID)
	}
}
