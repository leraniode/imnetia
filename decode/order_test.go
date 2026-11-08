package decode

import (
	"reflect"
	"testing"

	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

// Fragments with identical scores must come back in the same order every run.
func TestAssemble_TiedScoresOrderedByID(t *testing.T) {
	ks := knowledge.New()
	var entries []knowledge.Entry
	for _, id := range []string{"c", "a", "b"} {
		entries = append(entries, knowledge.Entry{
			ID:        id,
			Prototype: core.NewPrototype(core.SparseVector{1: 1.0}, 0.8),
			Surface:   "fragment " + id,
		})
	}
	if err := ks.Load("d", knowledge.Foundation, entries); err != nil {
		t.Fatal(err)
	}
	sig := core.NewSignalFromVector(core.SparseVector{1: 1.0}, 1.0)

	want := []string{"fragment a", "fragment b", "fragment c"}
	for i := 0; i < 50; i++ {
		resp, err := Assemble(sig, ks, "d", 0.5, 0.1)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(resp.Fragments, want) {
			t.Fatalf("tied fragments must order by ID: got %v want %v", resp.Fragments, want)
		}
	}
}
