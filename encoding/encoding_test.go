package encoding

import (
	"testing"

	"github.com/leraniode/x/centrix/registry"
)

// buildRegistry builds a test Registry with known concept entries.
func buildRegistry() *registry.Registry {
	reg := registry.New()

	// Single concepts
	reg.ID("lang.python")
	reg.ID("lang.go")
	reg.ID("concept.string")
	reg.ID("concept.reverse")
	reg.ID("concept.file")
	reg.ID("concept.read")
	reg.ID("concept.write")
	reg.ID("concept.error")
	reg.ID("concept.handle")

	// Multi-word concepts
	reg.ID("list comprehension")
	reg.ID("nil pointer")
	reg.ID("error handling")

	return reg
}

func TestEncodeDirectMatch(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	vec := enc.Encode("python string reverse")

	if _, ok := vec[reg.ID("lang.python")]; !ok {
		t.Error("expected lang.python in vector")
	}
	if _, ok := vec[reg.ID("concept.string")]; !ok {
		t.Error("expected concept.string in vector")
	}
	if _, ok := vec[reg.ID("concept.reverse")]; !ok {
		t.Error("expected concept.reverse in vector")
	}
}

func TestEncodeBigramMatch(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	vec := enc.Encode("list comprehension in python")

	// "list comprehension" should match as bigram at full weight
	bigramIdx := reg.ID("list comprehension")
	if w, ok := vec[bigramIdx]; !ok {
		t.Error("expected 'list comprehension' bigram in vector")
	} else if w != weightDirect {
		t.Errorf("expected bigram weight %v, got %v", weightDirect, w)
	}

	// python should match as unigram
	if _, ok := vec[reg.ID("lang.python")]; !ok {
		t.Error("expected lang.python in vector")
	}
}

func TestEncodeTrigramMatch(t *testing.T) {
	reg := buildRegistry()
	// Add a trigram
	reg.ID("read a file")
	enc := New(reg)

	vec := enc.Encode("read a file in python")

	trigramIdx := reg.ID("read a file")
	if _, ok := vec[trigramIdx]; !ok {
		t.Error("expected 'read a file' trigram in vector")
	}
}

func TestEncodeStopWordsDropped(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	// "how do i" are all stop words
	vec := enc.Encode("how do i read a file")

	// Only "read" and "file" should match
	if _, ok := vec[reg.ID("concept.read")]; !ok {
		t.Error("expected concept.read in vector")
	}
	if _, ok := vec[reg.ID("concept.file")]; !ok {
		t.Error("expected concept.file in vector")
	}
	// Vector should be small — stop words dropped
	if len(vec) > 2 {
		t.Errorf("expected at most 2 features (stop words dropped), got %d", len(vec))
	}
}

func TestEncodeStemming(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	// "reversing" should stem to "reverse" → concept.reverse
	vec := enc.Encode("reversing a string")

	if _, ok := vec[reg.ID("concept.reverse")]; !ok {
		t.Error("expected concept.reverse via stemming 'reversing'")
	}
	// Stemmed weight should be less than direct
	if w, ok := vec[reg.ID("concept.reverse")]; ok {
		if w >= weightDirect {
			t.Errorf("expected stemmed weight < %v, got %v", weightDirect, w)
		}
	}
}

func TestEncodeUnknownTokensDropped(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	vec := enc.Encode("xyzzy florp quux")

	if len(vec) != 0 {
		t.Errorf("expected empty vector for all-unknown tokens, got %d features", len(vec))
	}
}

func TestEncodeEmptyString(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	vec := enc.Encode("")
	if len(vec) != 0 {
		t.Errorf("expected empty vector for empty input, got %d features", len(vec))
	}
}

func TestEncodeDeterministic(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	text := "read a file in python"
	v1 := enc.Encode(text)
	v2 := enc.Encode(text)

	if len(v1) != len(v2) {
		t.Fatal("encoding not deterministic — different vector lengths")
	}
	for idx, w1 := range v1 {
		w2, ok := v2[idx]
		if !ok || w1 != w2 {
			t.Fatalf("encoding not deterministic — feature %d: %v vs %v", idx, w1, w2)
		}
	}
}

func TestEncodePunctuationStripped(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	// Punctuation should be stripped — same result as without it
	v1 := enc.Encode("python string reverse")
	v2 := enc.Encode("python, string? reverse!")

	if len(v1) != len(v2) {
		t.Errorf("punctuation not stripped properly: %d vs %d features", len(v1), len(v2))
	}
}

func TestDecode(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	vec := enc.Encode("python file read")
	names := enc.Decode(vec)

	if len(names) == 0 {
		t.Fatal("expected non-empty names from Decode")
	}
	// All names should be non-empty strings
	for _, n := range names {
		if n == "" {
			t.Error("expected non-empty name from Decode")
		}
	}
}

func TestEncodeNilPointerBigram(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	vec := enc.Encode("nil pointer error in go")

	// "nil pointer" should be bigram
	if _, ok := vec[reg.ID("nil pointer")]; !ok {
		t.Error("expected 'nil pointer' bigram match")
	}
	// "go" → lang.go
	if _, ok := vec[reg.ID("lang.go")]; !ok {
		t.Error("expected lang.go in vector")
	}
}
