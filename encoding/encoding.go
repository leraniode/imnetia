// Package encoding converts raw text into SparseVectors using the Registry.
//
// This is the runtime encoding path — what happens when a user types a query
// and the ContextEngine needs to turn it into a Signal before the flow runs.
//
// The approach: Registry-based tokenisation.
//   - The Registry is the vocabulary — every concept the system understands
//     has a Registry entry mapped to a stable FeatureIndex.
//   - Words not in the Registry are unknown — they contribute nothing.
//     This is honest: the system doesn't know that word, so it has no features.
//   - Multi-word concepts (bigrams, trigrams) are tried before single words.
//   - Basic suffix stripping handles common inflections.
//   - Synonyms are deferred to v2.
//
// The Encoder does not produce embeddings. It does not use statistical models.
// It is a deterministic lookup — same input always produces the same Vector.
package encoding

import (
	"strings"
	"unicode"

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

	// weightUnigram is the weight penalty when a unigram matches but a longer
	// ngram containing it also matched. Applied when the unigram is "consumed"
	// by a bigram or trigram — we still include it but with reduced weight.
	weightUnigram = 0.6
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

// suffixRules maps suffix → stripped form.
// Applied in order — longest suffix first matters.
// Only strip if the resulting stem is at least 3 characters.
var suffixRules = []struct {
	suffix  string
	replace string
}{
	{"tion", ""},   // generation → generat (approximate, but hits "generate")
	{"ing", ""},    // reversing → revers → reverse after e-restore
	{"ness", ""},   // friendliness → friendli → friendly
	{"ly", ""},     // quickly → quick
	{"ed", ""},     // reversed → revers
	{"er", ""},     // handler → handl
	{"est", ""},    // fastest → fast
	{"s", ""},      // strings → string (applied last, most aggressive)
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
		if e.reg.Has(phrase) {
			idx := e.reg.ID(phrase)
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
		if e.reg.Has(phrase) {
			idx := e.reg.ID(phrase)
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

		// Direct Registry lookup — Has before ID to avoid auto-registration.
		if e.reg.Has(token) {
			accumulate(vec, e.reg.ID(token), weightDirect)
			continue
		}

		// Suffix stripping fallback.
		if stemmed, ok := e.tryStem(token); ok {
			if e.reg.Has(stemmed) {
				accumulate(vec, e.reg.ID(stemmed), weightStemmed)
			}
		}
		// No hit: dropped. Honest about unknown tokens.
	}

	return vec
}

// Decode converts a SparseVector back into a list of concept names.
// Uses the Registry to map FeatureIndexes → human-readable names.
// Features with no Registry entry are shown as "?<index>".
// Useful for debugging and trace readability.
func (e *Encoder) Decode(vec core.SparseVector) []string {
	names := make([]string, 0, len(vec))
	for idx := range vec {
		name, ok := e.reg.Name(idx)
		if !ok {
			name = "?"
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

// tryStem applies suffix stripping rules and returns the stemmed form.
// Returns "", false if no rule applies or the stem is too short.
func (e *Encoder) tryStem(token string) (string, bool) {
	for _, rule := range suffixRules {
		if !strings.HasSuffix(token, rule.suffix) {
			continue
		}
		stem := token[:len(token)-len(rule.suffix)]

		// Minimum stem length.
		if len(stem) < 3 {
			continue
		}

		// Try stem as-is.
		if e.reg.Has(stem) {
			return stem, true
		}

		// Try stem + "e" restoration (reversing → reverse).
		if e.reg.Has(stem + "e") {
			return stem + "e", true
		}
	}
	return "", false
}

// accumulate adds weight to a feature in the vector.
// If the feature already exists, takes the max (don't double-count).
func accumulate(vec core.SparseVector, idx core.FeatureIndex, weight float64) {
	if existing, ok := vec[idx]; !ok || weight > existing {
		vec[idx] = weight
	}
}
