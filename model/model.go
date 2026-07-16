// Package model packages a Flow, KnowledgeStore, and learned state into a
// portable model artifact (.ilm file) that can be saved, loaded, and run
// by the Engine or served by Onix.
//
// What a model artifact contains:
//   - Metadata: name, version, description, created timestamp
//   - Flow structure: nodes (IDs only — handlers are re-registered at load time),
//     edges with current Weight and Trust values
//   - Intelligence knowledge: Prototype IDs and current weights for all
//     Intelligence domains. Foundation Prototypes are not stored here —
//     they are loaded from .pack files separately at startup.
//
// What a model artifact does NOT contain:
//   - NodeHandler functions (Go functions cannot be serialised)
//   - Foundation Prototype content (loaded from .pack files)
//   - The registry (shared, not per-model)
//
// Loading workflow:
//  1. Developer loads Foundation .pack files into a KnowledgeStore
//  2. Developer calls model.Load() to restore learned state on top
//  3. Developer re-registers NodeHandlers onto the restored nodes
//  4. Pass Flow and Store to engine.New() — ready to run
//
// The .ilm format is JSON. Simple, inspectable, version-controlled friendly.
package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

// Extension for Imnetia model artifact files.
const Extension = ".ilm"

// currentVersion is the .ilm format version. Increment on breaking changes.
const currentVersion = 1

// ─── Errors ───────────────────────────────────────────────────────────────────

var (
	ErrVersionMismatch = errors.New("imnetia/model: unsupported .ilm format version")
	ErrNoName          = errors.New("imnetia/model: model name must not be empty")
)

// ─── Artifact types ───────────────────────────────────────────────────────────

// Artifact is the serialisable model state.
type Artifact struct {
	// Metadata
	FormatVersion int       `json:"format_version"`
	Name          string    `json:"name"`
	Version       string    `json:"version"`
	Description   string    `json:"description,omitempty"`
	CreatedAt     time.Time `json:"created_at"`

	// Flow structure
	Nodes []nodeRecord `json:"nodes"`
	Edges []edgeRecord `json:"edges"`
	Entry string       `json:"entry"`

	// Learned state — Intelligence Prototypes only
	IntelligenceWeights []protoWeight `json:"intelligence_weights,omitempty"`
}

type nodeRecord struct {
	ID string `json:"id"`
}

type edgeRecord struct {
	From   string  `json:"from"`
	To     string  `json:"to"`
	Weight float64 `json:"weight"`
	Trust  float64 `json:"trust"`
}

type protoWeight struct {
	Domain string  `json:"domain"`
	ID     string  `json:"id"`
	Weight float64 `json:"weight"`
}

// ─── Build ────────────────────────────────────────────────────────────────────

// Build creates an Artifact from the current state of a Flow and KnowledgeStore.
//
// name and version identify the model. description is optional.
//
// The resulting Artifact captures:
//   - All node IDs from the Flow (handlers are not serialisable)
//   - All edge Weight and Trust values
//   - Current weights for all Intelligence domain Prototypes
func Build(
	flow *graph.Flow,
	store *knowledge.KnowledgeStore,
	name, version, description string,
) (*Artifact, error) {
	if name == "" {
		return nil, ErrNoName
	}

	art := &Artifact{
		FormatVersion: currentVersion,
		Name:          name,
		Version:       version,
		Description:   description,
		CreatedAt:     time.Now().UTC(),
		Entry:         flow.EntryID(),
	}

	// Capture node IDs.
	for _, id := range flow.Nodes() {
		art.Nodes = append(art.Nodes, nodeRecord{ID: id})
	}

	// Capture edges with current Weight and Trust.
	for _, id := range flow.Nodes() {
		for _, e := range flow.OutEdges(id) {
			art.Edges = append(art.Edges, edgeRecord{
				From:   e.From,
				To:     e.To,
				Weight: e.Weight,
				Trust:  e.Trust,
			})
		}
	}

	// Capture Intelligence Prototype weights.
	for _, domain := range store.Domains() {
		entries, kind, err := store.Snapshot(domain)
		if err != nil || kind != knowledge.Intelligence {
			continue
		}
		for _, e := range entries {
			art.IntelligenceWeights = append(art.IntelligenceWeights, protoWeight{
				Domain: domain,
				ID:     e.ID,
				Weight: e.Prototype.Weight,
			})
		}
	}

	return art, nil
}

// ─── Save / Load ──────────────────────────────────────────────────────────────

// Save writes an Artifact to path as a JSON .ilm file.
func Save(art *Artifact, path string) error {
	data, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		return fmt.Errorf("imnetia/model: marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("imnetia/model: write %q: %w", path, err)
	}
	return nil
}

// LoadArtifact reads an .ilm file from path and returns its Artifact.
func LoadArtifact(path string) (*Artifact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("imnetia/model: read %q: %w", path, err)
	}
	var art Artifact
	if err := json.Unmarshal(data, &art); err != nil {
		return nil, fmt.Errorf("imnetia/model: unmarshal: %w", err)
	}
	if art.FormatVersion != currentVersion {
		return nil, fmt.Errorf("%w: got %d, want %d",
			ErrVersionMismatch, art.FormatVersion, currentVersion)
	}
	return &art, nil
}

// ─── Restore ──────────────────────────────────────────────────────────────────

// RestoreFlow rebuilds a Flow from an Artifact.
//
// IMPORTANT: Node handlers are not serialised. After RestoreFlow, the caller
// must re-register handlers onto each node before running the flow:
//
//	flow, err := model.RestoreFlow(art)
//	node, _ := flow.Node("classify")
//	node.Handler = myClassifyHandler
//
// Edges with Weight and Trust are restored from the Artifact.
// The entry node is set automatically.
func RestoreFlow(art *Artifact) (*graph.Flow, error) {
	f := graph.NewFlow()

	for _, n := range art.Nodes {
		node, err := graph.NewNode(n.ID, nil)
		if err != nil {
			return nil, fmt.Errorf("imnetia/model: restore node %q: %w", n.ID, err)
		}
		if err := f.Add(node); err != nil {
			return nil, fmt.Errorf("imnetia/model: add node %q: %w", n.ID, err)
		}
	}

	for _, e := range art.Edges {
		if err := f.Link(e.From, e.To, e.Weight, e.Trust); err != nil {
			return nil, fmt.Errorf("imnetia/model: restore edge %q→%q: %w", e.From, e.To, err)
		}
	}

	if art.Entry != "" {
		if err := f.Entry(art.Entry); err != nil {
			return nil, fmt.Errorf("imnetia/model: set entry %q: %w", art.Entry, err)
		}
	}

	return f, nil
}

// RestoreWeights applies Intelligence Prototype weights from an Artifact
// back onto a KnowledgeStore.
//
// The KnowledgeStore must already have the relevant Intelligence domains
// loaded (via Load or Register) before calling this. Foundation domains
// are never written to — they are loaded from .pack files separately.
//
// Prototype IDs not found in the store are skipped silently.
func RestoreWeights(art *Artifact, store *knowledge.KnowledgeStore) {
	for _, pw := range art.IntelligenceWeights {
		_ = store.UpdateWeight(pw.Domain, pw.ID, pw.Weight)
	}
}

// ─── Convenience ──────────────────────────────────────────────────────────────

// Info returns a one-line summary of an Artifact for display.
func Info(art *Artifact) string {
	return fmt.Sprintf("%s v%s — %d nodes, %d edges, %d learned weights (created %s)",
		art.Name,
		art.Version,
		len(art.Nodes),
		len(art.Edges),
		len(art.IntelligenceWeights),
		art.CreatedAt.Format("2006-01-02"),
	)
}

// NewPrototypeFrom is a helper to build a knowledge.Entry for use when
// populating an Intelligence domain that will be managed by this package.
// It ensures the Prototype is constructed correctly for the store.
func NewPrototypeFrom(id string, vec core.SparseVector, weight float64) knowledge.Entry {
	return knowledge.Entry{
		ID:        id,
		Prototype: core.NewPrototype(vec, weight),
	}
}
