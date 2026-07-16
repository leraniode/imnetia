package model

import (
	"os"
	"testing"

	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

func buildFlow() *graph.Flow {
	f := graph.NewFlow()
	a, _ := graph.NewNode("classify", nil)
	b, _ := graph.NewNode("respond", nil)
	f.Add(a)
	f.Add(b)
	f.Link("classify", "respond", 0.8, 0.7)
	f.Entry("classify")
	return f
}

func buildStore() *knowledge.KnowledgeStore {
	ks := knowledge.New()
	ks.Load("intent", knowledge.Intelligence, []knowledge.Entry{
		{ID: "intent.greet", Prototype: core.NewPrototype(core.SparseVector{1: 1.0}, 0.6)},
	})
	ks.Load("facts", knowledge.Foundation, []knowledge.Entry{
		{ID: "fact.one", Prototype: core.NewPrototype(core.SparseVector{2: 1.0}, 1.0)},
	})
	return ks
}

func TestBuild(t *testing.T) {
	f := buildFlow()
	ks := buildStore()

	art, err := Build(f, ks, "test-model", "0.1.0", "a test model")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if art.Name != "test-model" {
		t.Fatalf("expected name 'test-model', got %q", art.Name)
	}
	if len(art.Nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(art.Nodes))
	}
	if len(art.Edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(art.Edges))
	}
	if art.Entry != "classify" {
		t.Fatalf("expected entry 'classify', got %q", art.Entry)
	}
}

func TestBuildExcludesFoundation(t *testing.T) {
	f := buildFlow()
	ks := buildStore()

	art, _ := Build(f, ks, "test-model", "0.1.0", "")
	for _, pw := range art.IntelligenceWeights {
		if pw.Domain == "facts" {
			t.Fatal("Foundation domain should not appear in artifact")
		}
	}
}

func TestBuildNoName(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	_, err := Build(f, ks, "", "0.1.0", "")
	if err == nil {
		t.Fatal("expected ErrNoName")
	}
}

func TestSaveAndLoad(t *testing.T) {
	f := buildFlow()
	ks := buildStore()

	art, _ := Build(f, ks, "test-model", "0.1.0", "")

	tmp, err := os.CreateTemp("", "model-*.ilm")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	tmp.Close()
	defer os.Remove(tmp.Name())

	if err := Save(art, tmp.Name()); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := LoadArtifact(tmp.Name())
	if err != nil {
		t.Fatalf("LoadArtifact: %v", err)
	}
	if loaded.Name != art.Name {
		t.Fatalf("name mismatch: %q vs %q", loaded.Name, art.Name)
	}
	if loaded.FormatVersion != currentVersion {
		t.Fatalf("version mismatch")
	}
	if len(loaded.Nodes) != 2 {
		t.Fatalf("expected 2 nodes after load, got %d", len(loaded.Nodes))
	}
}

func TestRestoreFlow(t *testing.T) {
	f := buildFlow()
	ks := buildStore()

	art, _ := Build(f, ks, "test-model", "0.1.0", "")
	restored, err := RestoreFlow(art)
	if err != nil {
		t.Fatalf("RestoreFlow: %v", err)
	}

	// Nodes should be restored
	if restored.Len() != 2 {
		t.Fatalf("expected 2 nodes, got %d", restored.Len())
	}

	// Edges with original weights
	edges := restored.OutEdges("classify")
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].Weight != 0.8 {
		t.Fatalf("expected weight 0.8, got %v", edges[0].Weight)
	}
	if edges[0].Trust != 0.7 {
		t.Fatalf("expected trust 0.7, got %v", edges[0].Trust)
	}

	// Entry set
	if restored.EntryID() != "classify" {
		t.Fatalf("expected entry 'classify', got %q", restored.EntryID())
	}

	// Handlers are nil — caller must re-register
	node, ok := restored.Node("classify")
	if !ok {
		t.Fatal("classify node not found")
	}
	if node.Handler != nil {
		t.Fatal("handler should be nil after restore")
	}
}

func TestRestoreWeights(t *testing.T) {
	f := buildFlow()
	ks := buildStore()

	// Change a weight
	ks.UpdateWeight("intent", "intent.greet", 0.9)

	art, _ := Build(f, ks, "test-model", "0.1.0", "")

	// Reset weight
	ks.UpdateWeight("intent", "intent.greet", 0.1)

	// Restore from artifact
	RestoreWeights(art, ks)

	e, err := ks.Get("intent", "intent.greet")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if e.Prototype.Weight != 0.9 {
		t.Fatalf("expected restored weight 0.9, got %v", e.Prototype.Weight)
	}
}

func TestInfo(t *testing.T) {
	f := buildFlow()
	ks := buildStore()
	art, _ := Build(f, ks, "test-model", "0.1.0", "")
	s := Info(art)
	if s == "" {
		t.Fatal("expected non-empty info string")
	}
}

func TestFullRoundTrip(t *testing.T) {
	// Build → Save → Load → RestoreFlow → RestoreWeights
	f := buildFlow()
	ks := buildStore()

	art, _ := Build(f, ks, "roundtrip", "1.0.0", "full round trip test")

	tmp, _ := os.CreateTemp("", "rt-*.ilm")
	tmp.Close()
	defer os.Remove(tmp.Name())

	Save(art, tmp.Name())

	loaded, err := LoadArtifact(tmp.Name())
	if err != nil {
		t.Fatalf("LoadArtifact: %v", err)
	}

	f2, err := RestoreFlow(loaded)
	if err != nil {
		t.Fatalf("RestoreFlow: %v", err)
	}
	RestoreWeights(loaded, ks)

	if f2.Len() != f.Len() {
		t.Fatalf("node count mismatch after full round trip")
	}
}
