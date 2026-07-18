package decode

import (
	"testing"

	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

func makeStore() *knowledge.KnowledgeStore {
	ks := knowledge.New()
	ks.Load("intent", knowledge.Foundation, []knowledge.Entry{
		{
			ID:        "intent.greet",
			Prototype: core.NewPrototype(core.SparseVector{1: 1.0, 2: 0.8}, 0.9),
		},
		{
			ID:        "intent.farewell",
			Prototype: core.NewPrototype(core.SparseVector{3: 1.0, 4: 0.7}, 0.8),
		},
	})
	return ks
}

func TestClassifyFindsMatch(t *testing.T) {
	ks := makeStore()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 2: 0.7}, 0.85)

	result, err := Classify(sig, ks, "intent", DefaultThreshold)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Match != "intent.greet" {
		t.Fatalf("expected intent.greet, got %q", result.Match)
	}
	if result.Score <= 0 {
		t.Fatal("expected positive score")
	}
	if !result.Certain {
		t.Fatalf("expected certain result (confidence=%.2f, threshold=%.2f)",
			result.Confidence, DefaultThreshold)
	}
}

func TestClassifyUncertain(t *testing.T) {
	ks := makeStore()
	// Low confidence signal
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9}, 0.3)

	result, err := Classify(sig, ks, "intent", DefaultThreshold)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Certain {
		t.Fatal("expected uncertain result for low confidence")
	}
}

func TestClassifyNoDomain(t *testing.T) {
	ks := makeStore()
	sig := core.NewSignalFromVector(core.SparseVector{1: 1.0}, 0.9)

	_, err := Classify(sig, ks, "", DefaultThreshold)
	if err == nil {
		t.Fatal("expected ErrNoDomain")
	}
}

func TestClassifyNoMatch(t *testing.T) {
	ks := makeStore()
	// Vector orthogonal to all prototypes
	sig := core.NewSignalFromVector(core.SparseVector{99: 1.0}, 0.9)

	result, err := Classify(sig, ks, "intent", DefaultThreshold)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Match != "" {
		t.Fatalf("expected no match, got %q", result.Match)
	}
}

func TestGate(t *testing.T) {
	high := core.NewSignalFromVector(core.SparseVector{1: 1.0}, 0.9)
	low := core.NewSignalFromVector(core.SparseVector{1: 1.0}, 0.2)

	if !Gate(high, DefaultThreshold) {
		t.Fatal("expected high-confidence signal to pass gate")
	}
	if Gate(low, DefaultThreshold) {
		t.Fatal("expected low-confidence signal to fail gate")
	}
}

func TestRunNoDecoder(t *testing.T) {
	ks := makeStore()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 2: 0.7}, 0.85)

	output, result, err := Run(sig, ks, "intent", DefaultThreshold, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if output == nil {
		t.Fatal("expected non-nil output")
	}
	if result.Match != "intent.greet" {
		t.Fatalf("expected greet match, got %q", result.Match)
	}
}

func TestRunNoDecoderLowConfidence(t *testing.T) {
	ks := makeStore()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9}, 0.1) // low confidence

	_, _, err := Run(sig, ks, "intent", DefaultThreshold, nil)
	if err == nil {
		t.Fatal("expected ErrLowConfidence")
	}
}

// testDecoder is a simple Decoder implementation for testing.
type testDecoder struct{}

func (d *testDecoder) Decode(signal core.Signal, result Result) (any, error) {
	if !result.Certain {
		return "unknown", nil
	}
	return result.Match, nil
}

func TestRunWithDecoder(t *testing.T) {
	ks := makeStore()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 2: 0.7}, 0.85)
	dec := &testDecoder{}

	output, result, err := Run(sig, ks, "intent", DefaultThreshold, dec)
	if err != nil {
		t.Fatalf("Run with decoder: %v", err)
	}
	if output != "intent.greet" {
		t.Fatalf("expected decoder to return intent.greet, got %v", output)
	}
	_ = result
}

func TestRunWithDecoderUncertain(t *testing.T) {
	ks := makeStore()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.1}, 0.1) // low confidence
	dec := &testDecoder{}

	output, _, err := Run(sig, ks, "intent", DefaultThreshold, dec)
	if err != nil {
		t.Fatalf("decoder should handle uncertainty: %v", err)
	}
	if output != "unknown" {
		t.Fatalf("expected decoder to return 'unknown', got %v", output)
	}
}

func TestSummarise(t *testing.T) {
	r := Result{
		Match:      "intent.greet",
		Domain:     "intent",
		Score:      0.82,
		Confidence: 0.87,
		Certain:    true,
	}
	s := Summarise(r)
	if s == "" {
		t.Fatal("expected non-empty summary")
	}
}

func TestRunNoDomain(t *testing.T) {
	ks := makeStore()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9}, 0.85)

	output, result, err := Run(sig, ks, "", DefaultThreshold, nil)
	if err != nil {
		t.Fatalf("Run with no domain: %v", err)
	}
	if result.Match != "" {
		t.Fatal("expected no match when domain is empty")
	}
	_ = output
}

// ─── Assemble tests ───────────────────────────────────────────────────────────

func makeStoreWithSurface() *knowledge.KnowledgeStore {
	ks := knowledge.New()
	ks.Load("python", knowledge.Foundation, []knowledge.Entry{
		{
			ID:        "python.string.reverse",
			Prototype: core.NewPrototype(core.SparseVector{1: 0.9, 2: 0.85, 3: 0.9}, 0.9),
			Surface:   "reverse a string in Python using slice notation: [::-1]",
		},
		{
			ID:        "python.string.reverse.example",
			Prototype: core.NewPrototype(core.SparseVector{1: 0.8, 2: 0.7, 3: 0.85, 4: 0.8}, 0.8),
			Surface:   "example: s = 'hello'; reversed_s = s[::-1]  # 'olleh'",
		},
		{
			ID:        "python.string.upper",
			Prototype: core.NewPrototype(core.SparseVector{1: 0.9, 5: 0.9}, 0.85),
			Surface:   "convert string to uppercase: s.upper()",
		},
		{
			ID:        "python.reasoning.only",
			Prototype: core.NewPrototype(core.SparseVector{1: 0.9, 6: 0.9}, 0.7),
			Surface:   "", // no surface — reasoning only
		},
	})
	return ks
}

func TestAssembleFindsFragments(t *testing.T) {
	ks := makeStoreWithSurface()
	// Signal close to string reverse
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 2: 0.8, 3: 0.9}, 0.85)

	resp, err := Assemble(sig, ks, "python", DefaultThreshold, 0.1)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(resp.Fragments) == 0 {
		t.Fatal("expected fragments in assembled response")
	}
	// Best match should be the reverse entry
	if resp.Fragments[0] == "" {
		t.Fatal("expected non-empty first fragment")
	}
}

func TestAssembleEmptySurfaceSkipped(t *testing.T) {
	ks := makeStoreWithSurface()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 6: 0.9}, 0.85)

	resp, err := Assemble(sig, ks, "python", DefaultThreshold, 0.1)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// "python.reasoning.only" has empty Surface — should never appear
	for _, f := range resp.Fragments {
		if f == "" {
			t.Fatal("empty surface should be skipped in fragments")
		}
	}
}

func TestAssembleOrderedByScore(t *testing.T) {
	ks := makeStoreWithSurface()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 2: 0.85, 3: 0.9}, 0.85)

	resp, err := Assemble(sig, ks, "python", DefaultThreshold, 0.1)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Scores should be descending
	for i := 1; i < len(resp.Scores); i++ {
		if resp.Scores[i] > resp.Scores[i-1] {
			t.Fatalf("fragments not sorted by score: [%d]=%v > [%d]=%v",
				i, resp.Scores[i], i-1, resp.Scores[i-1])
		}
	}
}

func TestAssembleHighThresholdFilters(t *testing.T) {
	ks := makeStoreWithSurface()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 2: 0.85, 3: 0.9}, 0.85)

	// Very high score threshold — should filter most fragments
	resp, err := Assemble(sig, ks, "python", DefaultThreshold, 0.99)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(resp.Fragments) > 1 {
		t.Fatalf("expected 0-1 fragments at high threshold, got %d", len(resp.Fragments))
	}
}

func TestAssembleCarriesResult(t *testing.T) {
	ks := makeStoreWithSurface()
	sig := core.NewSignalFromVector(core.SparseVector{1: 0.9, 2: 0.85, 3: 0.9}, 0.85)

	resp, err := Assemble(sig, ks, "python", DefaultThreshold, 0.1)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	// Result should be populated
	if resp.Result.Confidence == 0 {
		t.Fatal("expected non-zero confidence in Result")
	}
}
