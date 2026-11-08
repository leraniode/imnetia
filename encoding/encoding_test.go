package encoding

import (
	"errors"
	"reflect"
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

	// Surface forms: the words people type, mapped onto the namespaced concepts.
	// "python" → lang.python, "reverse" → concept.reverse, and so on.
	err := reg.AliasLastSegments(
		"lang.python", "lang.go", "concept.string", "concept.reverse", "concept.file",
		"concept.read", "concept.write", "concept.error", "concept.handle",
	)
	if err != nil {
		panic(err)
	}

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

// ─── Aliases ──────────────────────────────────────────────────────────────────

func TestAlias_NormalisesAndMatches(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	// Mixed case and punctuation are normalised to what Encode sees.
	if err := enc.Alias("lang.python", "Py", "Python 3!"); err != nil {
		t.Fatalf("Alias: %v", err)
	}

	vec := enc.Encode("py")
	if w, ok := vec[reg.ID("lang.python")]; !ok || w != weightDirect {
		t.Errorf("alias 'py' should hit lang.python at direct weight, got %v, %v", w, ok)
	}

	// A multi-word alias is matched as a bigram.
	vec = enc.Encode("how to use python 3 here")
	if _, ok := vec[reg.ID("lang.python")]; !ok {
		t.Error("multi-word alias 'python 3' should hit lang.python")
	}
}

func TestAlias_StemmedTokenReachesAlias(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	// "handling" → stem "handl" → "handle" → alias for concept.handle.
	vec := enc.Encode("handling")
	w, ok := vec[reg.ID("concept.handle")]
	if !ok {
		t.Fatal("expected concept.handle via stemmed alias")
	}
	if w >= weightDirect {
		t.Errorf("stemmed weight should be < %v, got %v", weightDirect, w)
	}
}

func TestAlias_Errors(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	if err := enc.Alias("lang.nope", "nope"); !errors.Is(err, registry.ErrUnknownConcept) {
		t.Errorf("unknown concept: want ErrUnknownConcept, got %v", err)
	}
	if err := enc.Alias("lang.go", "the"); err == nil {
		t.Error("a single stop word can never match and must be rejected")
	}
	if err := enc.Alias("lang.go", "  !! "); err == nil {
		t.Error("a surface with no tokens must be rejected")
	}
	// "error" is already a surface for concept.error; it cannot also mean lang.go.
	if err := enc.Alias("lang.go", "error"); !errors.Is(err, registry.ErrAliasConflict) {
		t.Errorf("conflict: want ErrAliasConflict, got %v", err)
	}
}

// Encode must never grow the feature space: the Registry is append-only, and
// every unknown token that registered itself would be a permanent feature.
func TestEncode_NeverRegisters(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)
	before, aliasesBefore := reg.Len(), reg.AliasLen()

	enc.Encode("xyzzy florp quux python reversing errors handlers")

	if reg.Len() != before || reg.AliasLen() != aliasesBefore {
		t.Errorf("Encode changed the Registry: concepts %d→%d, aliases %d→%d",
			before, reg.Len(), aliasesBefore, reg.AliasLen())
	}
}

// ─── Stemming ─────────────────────────────────────────────────────────────────

func TestStemming_Plurals(t *testing.T) {
	reg := buildRegistry()
	reg.ID("query")
	reg.ID("class")
	enc := New(reg)

	for text, concept := range map[string]string{
		"queries": "query", // ies → y
		"classes": "class", // es
		"files":   "concept.file",
		"errors":  "concept.error", // plain s
	} {
		vec := enc.Encode(text)
		if _, ok := vec[reg.ID(concept)]; !ok {
			t.Errorf("Encode(%q): expected %s", text, concept)
		}
	}
}

// ─── Decode ───────────────────────────────────────────────────────────────────

func TestDecode_OrderIsDeterministic(t *testing.T) {
	reg := buildRegistry()
	enc := New(reg)

	vec := enc.Encode("python file read string reverse error handle write")
	first := enc.Decode(vec)
	if len(first) < 5 {
		t.Fatalf("test needs a vector with several features, got %d", len(first))
	}
	for i := 0; i < 200; i++ {
		if got := enc.Decode(vec); !reflect.DeepEqual(got, first) {
			t.Fatalf("Decode order changed between calls:\n%v\n%v", first, got)
		}
	}
	// Ascending FeatureIndex order = registration order.
	if first[0] != "lang.python" {
		t.Errorf("expected ascending-index order starting at lang.python, got %v", first)
	}
}

func TestDecode_UnknownIndexShowsIndex(t *testing.T) {
	enc := New(buildRegistry())
	names := enc.Decode(map[uint32]float64{9999: 1.0})
	if len(names) != 1 || names[0] != "?9999" {
		t.Errorf("want [?9999], got %v", names)
	}
}
