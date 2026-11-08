// Package encoding converts raw text into SparseVectors using the Registry.
//
// This is the runtime encoding path — what happens when a user types a query
// and the ContextEngine needs to turn it into a Signal before the flow runs.
//
// The approach: Registry-based tokenisation.
//   - The Registry is the vocabulary — every concept the system understands
//     has a Registry entry mapped to a stable FeatureIndex.
//   - Concepts keep namespaced names ("lang.python"); the words people type
//     ("python", "py") are surface forms registered as aliases. The encoder
//     looks tokens up with Registry.Resolve, which finds either one and never
//     registers anything, so input text cannot grow the feature space.
//   - Words that resolve to nothing are unknown — they contribute nothing.
//     This is honest: the system doesn't know that word, so it has no features.
//   - Multi-word concepts (bigrams, trigrams) are tried before single words.
//   - Basic suffix stripping handles common inflections.
//   - Synonyms are deferred to v2.
//
// The Encoder does not produce embeddings. It does not use statistical models.
// It is a deterministic lookup — same input always produces the same Vector.
package encoding

import (
	"fmt"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/leraniode/x/centrix/core"
	"github.com/leraniode/x/centrix/registry"
)

// ─── Weights ──────────────────────────────────────────────────────────────────

const (
	// weightDirect is the feature weight for a direct Registry match.
	weightDirect = 1.0

	// weightStemmed is the feature weight when a match required suffix stripping.
	// Lower than direct — the stem is an approximation of the original token.
	weightStemmed = 0.8
)

// ─── Stop words ───────────────────────────────────────────────────────────────

// stopWords are tokens that carry no semantic meaning and are dropped.
// Keep this minimal — dropping too many words loses signal.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true,
	"is": true, "are": true, "was": true, "were": true,
	"be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true,
	"do": true, "does": true, "did": true,
	"will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "must": true, "shall": true,
	"i": true, "me": true, "my": true, "we": true, "us": true, "our": true,
	"you": true, "your": true, "it": true, "its": true,
	"this": true, "that": true, "these": true, "those": true,
	"in": true, "on": true, "at": true, "to": true, "for": true,
	"of": true, "with": true, "by": true, "from": true, "into": true,
	"and": true, "or": true, "but": true, "not": true, "no": true,
	"if": true, "then": true, "else": true, "when": true, "where": true,
	"what": true, "which": true, "who": true, "whom": true, "whose": true,
	"how": true, "why": true, "there": true, "here": true,
}

// ─── Suffixes ─────────────────────────────────────────────────────────────────

// suffixRules lists suffix → replacement, tried in order; the first rule whose
// result resolves wins. Only strip if the resulting stem is at least 3 runes.
// After each rule the stem is tried as-is, then with "e" restored.
var suffixRules = []struct {
	suffix  string
	replace string
}{
	{"ies", "y"}, // queries → query
	{"tion", ""}, // generation → generat (approximate; "generate" via e-restore)
	{"ing", ""},  // reversing → revers → reverse after e-restore
	{"ness", ""}, // quickness → quick
	{"ly", ""},   // quickly → quick
	{"ed", ""},   // reversed → revers → reverse
	{"er", ""},   // handler → handl → handle
	{"est", ""},  // fastest → fast
	{"es", ""},   // classes → class, files → fil → file
	{"s", ""},    // strings → string (last, most aggressive)
}

// ─── Encoder ──────────────────────────────────────────────────────────────────

// Encoder converts raw text into SparseVectors using the Registry.
// It is safe for concurrent use — the underlying Registry is read-only
// during encoding.
type Encoder struct {
	reg *registry.Registry
}

// New creates an Encoder backed by the given Registry.
// The Registry must contain all concept entries for the system's feature space.
func New(reg *registry.Registry) *Encoder {
	return &Encoder{reg: reg}
}

// Alias registers surface forms — words or phrases as they appear in input
// text — for a concept that is already in the Registry:
//
//	enc.Alias("lang.python", "python", "py")
//
// Each surface is normalised with the same tokeniser Encode uses (lower-case,
// punctuation stripped, single spaces), so "Python 3!" is stored as "python 3"
// and will match what Encode actually sees. A surface with no tokens, or a
// single stop word (which Encode would skip), is rejected. Surfaces are applied
// in order and stop at the first error; earlier ones stay registered.
//
// Errors from the Registry (registry.ErrUnknownConcept, registry.ErrAliasConflict)
// are returned unchanged and can be tested with errors.Is.
func (e *Encoder) Alias(concept string, surfaces ...string) error {
	norm := make([]string, len(surfaces))
	for i, s := range surfaces {
		tokens := tokenise(s)
		if len(tokens) == 0 {
			return fmt.Errorf("encoding: surface %q has no tokens", s)
		}
		if len(tokens) == 1 && stopWords[tokens[0]] {
			return fmt.Errorf("encoding: surface %q is a stop word and would never match", s)
		}
		norm[i] = strings.Join(tokens, " ")
	}
	for _, s := range norm {
		if err := e.reg.Alias(s, concept); err != nil {
			return err
		}
	}
	return nil
}

// Encode converts raw text into a SparseVector.
//
// Pipeline:
//  1. Normalise: lowercase, strip punctuation
//  2. Split into tokens
//  3. Try trigrams → bigrams → unigrams, longest match wins
//  4. For unigram misses: try suffix stripping
//  5. Build SparseVector from all hits
//
// Unknown tokens (no Registry entry after all attempts) are dropped.
// Same input always produces the same SparseVector — fully deterministic.
func (e *Encoder) Encode(text string) core.SparseVector {
	tokens := tokenise(text)
	if len(tokens) == 0 {
		return core.SparseVector{}
	}

	vec := core.SparseVector{}
	consumed := make([]bool, len(tokens))

	// Pass 1: try trigrams
	for i := 0; i <= len(tokens)-3; i++ {
		if consumed[i] {
			continue
		}
		phrase := tokens[i] + " " + tokens[i+1] + " " + tokens[i+2]
		if idx, ok := e.reg.Resolve(phrase); ok {
			accumulate(vec, idx, weightDirect)
			consumed[i] = true
			consumed[i+1] = true
			consumed[i+2] = true
		}
	}

	// Pass 2: try bigrams
	for i := 0; i <= len(tokens)-2; i++ {
		if consumed[i] || consumed[i+1] {
			continue
		}
		phrase := tokens[i] + " " + tokens[i+1]
		if idx, ok := e.reg.Resolve(phrase); ok {
			accumulate(vec, idx, weightDirect)
			consumed[i] = true
			consumed[i+1] = true
		}
	}

	// Pass 3: unigrams — unconsumed tokens
	for i, token := range tokens {
		if consumed[i] {
			continue
		}
		if stopWords[token] {
			continue
		}

		// Direct lookup — a concept name or an alias. Resolve never registers.
		if idx, ok := e.reg.Resolve(token); ok {
			accumulate(vec, idx, weightDirect)
			continue
		}

		// Suffix stripping fallback.
		if idx, ok := e.tryStem(token); ok {
			accumulate(vec, idx, weightStemmed)
		}
		// No hit: dropped. Honest about unknown tokens.
	}

	return vec
}

// Decode converts a SparseVector back into a list of concept names.
// Uses the Registry to map FeatureIndexes → human-readable names.
// Names come back in ascending FeatureIndex order, so the result is the same on
// every call. Features with no Registry entry appear as "?<index>".
// Useful for debugging and trace readability.
func (e *Encoder) Decode(vec core.SparseVector) []string {
	idxs := make([]core.FeatureIndex, 0, len(vec))
	for idx := range vec {
		idxs = append(idxs, idx)
	}
	slices.Sort(idxs)

	names := make([]string, 0, len(idxs))
	for _, idx := range idxs {
		name, ok := e.reg.Name(idx)
		if !ok {
			name = fmt.Sprintf("?%d", idx)
		}
		names = append(names, name)
	}
	return names
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// tokenise normalises text and splits into lowercase tokens.
// Strips punctuation, collapses whitespace.
func tokenise(text string) []string {
	// Lowercase.
	text = strings.ToLower(text)

	// Replace punctuation with spaces (preserve apostrophes in contractions).
	var b strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}

	// Split and filter empty strings.
	raw := strings.Fields(b.String())
	out := make([]string, 0, len(raw))
	for _, t := range raw {
		// Strip leading/trailing apostrophes.
		t = strings.Trim(t, "'")
		if len(t) > 0 {
			out = append(out, t)
		}
	}
	return out
}

// tryStem applies suffix stripping rules and returns the FeatureIndex of the
// first candidate stem that resolves. Returns 0, false if no rule applies,
// the stem is too short, or nothing resolves.
func (e *Encoder) tryStem(token string) (core.FeatureIndex, bool) {
	for _, rule := range suffixRules {
		if !strings.HasSuffix(token, rule.suffix) {
			continue
		}
		stem := token[:len(token)-len(rule.suffix)]

		// Minimum stem length, in runes.
		if utf8.RuneCountInString(stem) < 3 {
			continue
		}
		candidate := stem + rule.replace

		// Try the candidate as-is.
		if idx, ok := e.reg.Resolve(candidate); ok {
			return idx, true
		}

		// Try candidate + "e" restoration (reversing → reverse).
		if idx, ok := e.reg.Resolve(candidate + "e"); ok {
			return idx, true
		}
	}
	return 0, false
}

// accumulate adds weight to a feature in the vector.
// If the feature already exists, takes the max (don't double-count).
func accumulate(vec core.SparseVector, idx core.FeatureIndex, weight float64) {
	if existing, ok := vec[idx]; !ok || weight > existing {
		vec[idx] = weight
	}
}
