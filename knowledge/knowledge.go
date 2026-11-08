// Package knowledge manages the Prototype shelf that nodes consult during execution.
//
// A KnowledgeStore holds named Prototypes grouped by domain. Nodes query it
// by domain during a run to find what is relevant to their current Signal.
//
// Loading model:
//
//	Eager  — all domains loaded at store creation. Fast queries, higher memory.
//	         Correct for production serving where latency matters.
//	Lazy   — domains registered but not loaded until first query.
//	         Memory footprint scales with active domains, not total knowledge.
//	         Correct for constrained environments and embedded mode.
//
// The KnowledgeStore is safe for concurrent reads after all Load/Register
// calls are complete. It must not be written to during execution.
//
// Prototypes come from DotPack-produced .pack files. This package does not
// know about .pack format — it receives already-decoded Prototypes via Load.
// The connection between .pack files and the store is the model's responsibility.
package knowledge

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/leraniode/x/centrix/core"
)

// ─── Errors ───────────────────────────────────────────────────────────────────

var (
	// ErrDomainNotFound is returned when a queried domain has not been registered.
	ErrDomainNotFound = errors.New("imnetia/knowledge: domain not found")

	// ErrPrototypeNotFound is returned when a specific prototype ID is not found.
	ErrPrototypeNotFound = errors.New("imnetia/knowledge: prototype not found")

	// ErrDomainSealed is returned when attempting to add to a sealed domain.
	ErrDomainSealed = errors.New("imnetia/knowledge: domain is sealed (Foundation)")
)

// ─── DomainKind ───────────────────────────────────────────────────────────────

// DomainKind distinguishes Foundation (immutable) from Intelligence (evolving) domains.
// Foundation domains are sealed after loading — no Prototype may be added or
// modified. Intelligence domains can be updated by the learning system.
type DomainKind int

const (
	// Foundation — immutable after loading. Ground truth. Sealed on registration.
	Foundation DomainKind = iota

	// Intelligence — evolves through learning. Prototype weights update over time.
	Intelligence
)

// ─── Entry ────────────────────────────────────────────────────────────────────

// Entry is a named Prototype in the store.
// ID is stable within a domain — it identifies the knowledge unit across runs.
//
// Surface is the raw text fragment associated with this knowledge unit.
// It is used by the decode layer to assemble human-readable responses.
// The Vector is for reasoning (Cosine matching) — Surface is for output.
// Surface may be empty for knowledge units used purely for reasoning.
//
// One Prototype may have multiple Entries with different Surface texts —
// different ways of expressing the same concept in different contexts.
type Entry struct {
	ID        string
	Prototype core.Prototype
	Surface   string
}

// ─── domain ───────────────────────────────────────────────────────────────────

// domain is the internal representation of a loaded knowledge domain.
type domain struct {
	kind    DomainKind
	entries []Entry
	index   map[string]int // ID → slice index for O(1) lookup
	sealed  bool           // true for Foundation after loading
}

func newDomain(kind DomainKind) *domain {
	return &domain{
		kind:  kind,
		index: make(map[string]int),
	}
}

func (d *domain) add(e Entry) error {
	if d.sealed {
		return ErrDomainSealed
	}
	if _, exists := d.index[e.ID]; exists {
		return fmt.Errorf("imnetia/knowledge: duplicate entry ID %q in domain", e.ID)
	}
	idx := len(d.entries)
	d.entries = append(d.entries, e)
	d.index[e.ID] = idx
	return nil
}

func (d *domain) get(id string) (Entry, bool) {
	idx, ok := d.index[id]
	if !ok {
		return Entry{}, false
	}
	return d.entries[idx], true
}

func (d *domain) updateWeight(id string, weight float64) error {
	if d.sealed {
		return ErrDomainSealed
	}
	idx, ok := d.index[id]
	if !ok {
		return fmt.Errorf("%w: %q", ErrPrototypeNotFound, id)
	}
	d.entries[idx].Prototype.Weight = clamp01(weight)
	return nil
}

// ─── KnowledgeStore ───────────────────────────────────────────────────────────

// KnowledgeStore is the domain-scoped Prototype shelf.
//
// Nodes query it during execution to retrieve knowledge relevant to their
// current Signal. The store itself performs no similarity scoring —
// that is the runtime's job via Centrix algebra.
//
// Load mode controls when domains enter memory:
//   - In Eager mode, all registered domains are immediately available.
//   - In Lazy mode, domains are marked as registered but their Prototypes are
//     held by a loader function and only materialised on first Domain() call.
type KnowledgeStore struct {
	mu      sync.RWMutex
	domains map[string]*domain
	loaders map[string]func() ([]Entry, error) // lazy loaders, nil after load
	lazy    bool
}

// New creates a KnowledgeStore in Eager mode.
func New() *KnowledgeStore {
	return &KnowledgeStore{
		domains: make(map[string]*domain),
		loaders: make(map[string]func() ([]Entry, error)),
	}
}

// NewLazy creates a KnowledgeStore in Lazy mode.
// Domains are loaded on first access, not at registration time.
func NewLazy() *KnowledgeStore {
	ks := New()
	ks.lazy = true
	return ks
}

// Load registers a domain and adds all given entries to it.
// For Foundation domains, the domain is sealed immediately after loading —
// no further entries can be added.
//
// If the domain already exists, entries are appended (unless sealed).
// This allows incremental loading of Intelligence domains.
func (ks *KnowledgeStore) Load(domainName string, kind DomainKind, entries []Entry) error {
	ks.mu.Lock()
	defer ks.mu.Unlock()

	d, exists := ks.domains[domainName]
	if !exists {
		d = newDomain(kind)
		ks.domains[domainName] = d
	}

	for _, e := range entries {
		if err := d.add(e); err != nil {
			return fmt.Errorf("imnetia/knowledge: loading domain %q: %w", domainName, err)
		}
	}

	// Seal Foundation domains immediately — no future writes allowed.
	if kind == Foundation {
		d.sealed = true
	}

	return nil
}

// Register declares a domain with a lazy loader function.
// The loader is called on the first Domain() query for this domain.
// Panics if called on an Eager store — lazy registration requires NewLazy().
//
// This is the correct pattern for constrained environments: many domains can
// be registered upfront, but only the ones actually queried during a run
// will consume memory.
func (ks *KnowledgeStore) Register(domainName string, kind DomainKind, loader func() ([]Entry, error)) {
	if !ks.lazy {
		panic("imnetia/knowledge: Register requires a Lazy store — use NewLazy()")
	}
	ks.mu.Lock()
	defer ks.mu.Unlock()
	ks.loaders[domainName] = loader
	// Pre-create the domain shell so Has() works before load.
	if _, exists := ks.domains[domainName]; !exists {
		ks.domains[domainName] = newDomain(kind)
	}
}

// Domain returns all entries in the named domain, sorted by Prototype.Weight
// descending (highest trust first).
//
// In Lazy mode, the domain is loaded on first call. Subsequent calls are
// read-only and do not trigger loading.
//
// Returns ErrDomainNotFound if the domain has not been registered.
func (ks *KnowledgeStore) Domain(name string) ([]Entry, error) {
	// Fast path: read lock, return if already loaded.
	ks.mu.RLock()
	d, exists := ks.domains[name]
	if exists && (len(d.entries) > 0 || d.sealed) {
		entries := domainSnapshot(d)
		ks.mu.RUnlock()
		return entries, nil
	}
	ks.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("%w: %q", ErrDomainNotFound, name)
	}

	// Lazy load path: acquire write lock and materialise.
	if ks.lazy {
		ks.mu.Lock()
		defer ks.mu.Unlock()

		// Re-check after write lock — another goroutine may have loaded it.
		d = ks.domains[name]
		if len(d.entries) > 0 || d.sealed {
			return domainSnapshot(d), nil
		}

		loader, hasLoader := ks.loaders[name]
		if !hasLoader {
			return nil, fmt.Errorf("%w: %q (lazy, no loader registered)", ErrDomainNotFound, name)
		}

		loaded, err := loader()
		if err != nil {
			return nil, fmt.Errorf("imnetia/knowledge: lazy load %q: %w", name, err)
		}
		for _, e := range loaded {
			if err := d.add(e); err != nil {
				return nil, fmt.Errorf("imnetia/knowledge: lazy load %q: %w", name, err)
			}
		}
		if d.kind == Foundation {
			d.sealed = true
		}
		delete(ks.loaders, name) // loader consumed, free the closure
		return domainSnapshot(d), nil
	}

	// Eager store, domain exists but is empty — return empty slice, not error.
	return []Entry{}, nil
}

// Get returns a single Entry by domain and ID.
// Returns ErrDomainNotFound or ErrPrototypeNotFound if not found.
func (ks *KnowledgeStore) Get(domainName, id string) (Entry, error) {
	entries, err := ks.Domain(domainName)
	if err != nil {
		return Entry{}, err
	}
	// Linear scan — domains are typically small enough that this is fast.
	// If domain sizes grow large, an indexed lookup can be added here.
	for _, e := range entries {
		if e.ID == id {
			return e, nil
		}
	}
	return Entry{}, fmt.Errorf("%w: %q in domain %q", ErrPrototypeNotFound, id, domainName)
}

// UpdateWeight adjusts the Weight of a Prototype in an Intelligence domain.
// Called by the learning system after a run. Foundation domains reject updates.
func (ks *KnowledgeStore) UpdateWeight(domainName, id string, weight float64) error {
	ks.mu.Lock()
	defer ks.mu.Unlock()

	d, exists := ks.domains[domainName]
	if !exists {
		return fmt.Errorf("%w: %q", ErrDomainNotFound, domainName)
	}
	return d.updateWeight(id, weight)
}

// Has reports whether a domain with the given name is registered.
// Does not trigger a lazy load.
func (ks *KnowledgeStore) Has(domainName string) bool {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	_, ok := ks.domains[domainName]
	return ok
}

// Domains returns the names of all registered domains. Order is not guaranteed.
func (ks *KnowledgeStore) Domains() []string {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	names := make([]string, 0, len(ks.domains))
	for name := range ks.domains {
		names = append(names, name)
	}
	return names
}

// Size returns the total number of Prototype entries across all loaded domains.
// Unloaded lazy domains are not counted.
func (ks *KnowledgeStore) Size() int {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	total := 0
	for _, d := range ks.domains {
		total += len(d.entries)
	}
	return total
}

// Snapshot returns a copy of all entries in a domain, for persistence.
// Used by the model package when saving learned state.
// Returns ErrDomainNotFound if the domain is not registered.
func (ks *KnowledgeStore) Snapshot(domainName string) ([]Entry, DomainKind, error) {
	entries, err := ks.Domain(domainName)
	if err != nil {
		return nil, 0, err
	}
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	kind := ks.domains[domainName].kind
	return entries, kind, nil
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// domainSnapshot returns a copy of domain entries sorted by Weight descending.
// Highest-trust Prototypes surface first — nodes get the most reliable
// knowledge without needing to sort themselves.
// Called under a lock held by the caller.
func domainSnapshot(d *domain) []Entry {
	out := make([]Entry, len(d.entries))
	copy(out, d.entries)
	// Weight descending; equal weights fall back to ID so the order never
	// depends on the sort algorithm. Learning moves weights between runs, and
	// without a tie-break two tied entries could swap places for no reason.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Prototype.Weight != out[j].Prototype.Weight {
			return out[i].Prototype.Weight > out[j].Prototype.Weight
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
