// Package decode interprets the final Signal produced by Engine.Run.
//
// It has two layers:
//
//  1. Fixed core — always runs, Imnetia's own needs.
//     Confidence gating, best Prototype classification, Trace summary.
//     These run regardless of what model is built on top.
//
//  2. Extensible — the Decoder interface.
//     Models implement Decoder to turn a Result into their specific output:
//     a string response, a structured decision, a generated fragment.
//     The model's Decoder runs after the core layer, never instead of it.
//
// The confidence gate is non-negotiable. If the Signal's Confidence does
// not meet the threshold, the system surfaces honest uncertainty before
// the model's Decoder ever runs. This is the "no hallucination by
// construction" property at the decode layer.
package decode

import (
	"errors"
	"fmt"

	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

// ─── Constants ────────────────────────────────────────────────────────────────

// DefaultThreshold is the minimum Confidence required for a Result to be
// considered certain. Below this, Result.Certain is false.
// Based on the validated confidence constants from Centrix research:
// minMatchScore=0.15, so a threshold at 0.5 is a reasonable mid-range.
const DefaultThreshold = 0.50

// ─── Errors ───────────────────────────────────────────────────────────────────

var (
	// ErrLowConfidence is returned by Decode when the Signal's Confidence
	// is below threshold and no Decoder is provided to handle uncertainty.
	ErrLowConfidence = errors.New("imnetia/decode: signal confidence below threshold")

	// ErrNoDomain is returned when Classify is called with an empty domain.
	ErrNoDomain = errors.New("imnetia/decode: domain must not be empty")
)

// ─── Result ───────────────────────────────────────────────────────────────────

// Result is the fixed-core output of the decode layer.
// It is produced before any model-specific Decoder runs.
//
// Every run through Engine.Run produces a Result via the core layer.
// The model's Decoder then turns Result into its specific output type.
type Result struct {
	// Match is the ID of the best-matching Prototype in the queried domain.
	// Empty if no domain was queried or no positive match was found.
	Match string

	// Domain is the domain the Match came from.
	Domain string

	// Score is the combined similarity score: Cosine × Prototype.Weight.
	// [0.0, 1.0]. 0 if no match was found.
	Score float64

	// Confidence is the final Signal.Confidence after all node transformations.
	Confidence float64

	// Certain reports whether Confidence meets the threshold.
	// If false, the system does not know the answer well enough.
	// A model's Decoder should check this before producing output.
	Certain bool

	// Trace is the Signal's execution log — every step, every confidence delta.
	// Always present. The full reasoning record.
	Trace core.Trace
}

// ─── Decoder ──────────────────────────────────────────────────────────────────

// Decoder is the extensible interface models implement to turn a Result
// into their specific output.
//
// The core decode layer always runs first and produces a Result.
// The model's Decode method receives that Result and the original Signal,
// and returns whatever the model's output type is.
//
// If Result.Certain is false, Decode should return a meaningful
// "I don't know" response rather than guessing. What that looks like
// is the model's decision — the decode package does not dictate it.
type Decoder interface {
	Decode(signal core.Signal, result Result) (any, error)
}

// ─── Core layer ───────────────────────────────────────────────────────────────

// Classify finds the best-matching Prototype in the given domain for the Signal.
// Uses Cosine(signal.Vector, prototype.Vector) × prototype.Weight as the score.
//
// Returns a partially filled Result — Match, Domain, Score, Confidence, Certain.
// Trace is always set. The caller adds domain-specific fields if needed.
//
// Returns ErrNoDomain if domain is empty.
// Returns a Result with empty Match and Score=0 if no positive match is found.
func Classify(signal core.Signal, store *knowledge.KnowledgeStore, domain string, threshold float64) (Result, error) {
	if domain == "" {
		return Result{}, ErrNoDomain
	}

	result := Result{
		Confidence: signal.Confidence,
		Certain:    signal.Confidence >= threshold,
		Trace:      signal.Trace,
	}

	entries, err := store.Domain(domain)
	if err != nil || len(entries) == 0 {
		return result, nil // no match, not an error
	}

	var bestID string
	var bestScore float64

	for _, e := range entries {
		sim := core.Cosine(signal.Vector, e.Prototype.Vector)
		if sim <= 0 {
			continue
		}
		score := sim * e.Prototype.Weight
		if score > bestScore {
			bestScore = score
			bestID = e.ID
		}
	}

	result.Match = bestID
	result.Domain = domain
	result.Score = bestScore
	return result, nil
}

// Gate checks whether a Signal's Confidence meets the threshold.
// Returns true if the Signal is confident enough to act on.
// This is the minimum check — always run this before producing output.
func Gate(signal core.Signal, threshold float64) bool {
	return signal.Confidence >= threshold
}

// Summarise returns a human-readable summary of a Result for debugging.
func Summarise(r Result) string {
	certain := "uncertain"
	if r.Certain {
		certain = "certain"
	}
	if r.Match == "" {
		return fmt.Sprintf("[decode] confidence=%.2f (%s) no match in domain %q trace=%d steps",
			r.Confidence, certain, r.Domain, len(r.Trace.Steps()))
	}
	return fmt.Sprintf("[decode] confidence=%.2f (%s) match=%q score=%.3f domain=%q trace=%d steps",
		r.Confidence, certain, r.Match, r.Score, r.Domain, len(r.Trace.Steps()))
}

// ─── Decode ───────────────────────────────────────────────────────────────────

// Run runs the full decode pipeline:
//  1. Core layer: Classify produces a Result
//  2. Gate check: if uncertain and no Decoder, return ErrLowConfidence
//  3. Model layer: if a Decoder is provided, call it with Result
//
// domain is the Prototype domain to classify against. Pass "" to skip
// classification — Result will have empty Match but full Confidence and Trace.
//
// decoder may be nil — in that case, Run returns the Result directly as `any`,
// or ErrLowConfidence if confidence is below threshold.
func Run(
	signal core.Signal,
	store *knowledge.KnowledgeStore,
	domain string,
	threshold float64,
	decoder Decoder,
) (any, Result, error) {
	var result Result
	var err error

	if domain != "" {
		result, err = Classify(signal, store, domain, threshold)
		if err != nil {
			return nil, Result{}, err
		}
	} else {
		result = Result{
			Confidence: signal.Confidence,
			Certain:    signal.Confidence >= threshold,
			Trace:      signal.Trace,
		}
	}

	// No decoder — return Result directly.
	if decoder == nil {
		if !result.Certain {
			return nil, result, ErrLowConfidence
		}
		return result, result, nil
	}

	// Model decoder — always runs, even when uncertain.
	// The Decoder decides how to handle low confidence.
	output, err := decoder.Decode(signal, result)
	if err != nil {
		return nil, result, fmt.Errorf("imnetia/decode: Decoder.Decode: %w", err)
	}
	return output, result, nil
}
