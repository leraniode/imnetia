// Package session provides in-memory conversational continuity via
// Prototype Accumulation.
//
// Each conversation turn produces a final Signal. Rather than discarding it,
// the Store records its Vector as a temporary Prototype in a session-scoped
// KnowledgeStore. Subsequent nodes can query the session store the same way
// they query permanent knowledge — using ctx.Knowledge("session").
//
// Weights decay per turn — recent turns matter more than earlier ones.
// The session store never persists to disk. It lives for the duration of
// a conversation and is discarded afterward.
//
// Learning connection: at the end of a conversation, the session store's
// accumulated Signals form a complete training episode. The caller can
// pass the session to imnetia/learning to update Intelligence Prototypes
// based on what worked across the whole conversation.
package session

import (
	"fmt"

	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

const (
	// SessionDomain is the KnowledgeStore domain name used for session Prototypes.
	SessionDomain = "session"

	// DefaultDecay is the weight applied to prior turns.
	// Each turn's Signal is stored at Weight = DefaultDecay^(turns_ago).
	// Turn 0 (most recent) = 1.0, Turn 1 = 0.85, Turn 2 = 0.72, etc.
	DefaultDecay = 0.85
)

// Turn records one conversation turn — the final Signal and its turn index.
type Turn struct {
	Index  int
	Signal core.Signal
}

// Store is an in-memory session store.
// It accumulates Signals as Prototypes in a KnowledgeStore domain.
// Nodes query it via ctx.Knowledge(session.SessionDomain).
type Store struct {
	ks    *knowledge.KnowledgeStore
	turns []Turn
	decay float64
}

// New creates a Store with the given decay rate.
// Pass DefaultDecay for standard behaviour.
func New(decay float64) *Store {
	if decay <= 0 || decay >= 1 {
		decay = DefaultDecay
	}
	return &Store{
		ks:    knowledge.New(),
		decay: decay,
	}
}

// Record stores the final Signal from a completed turn.
// The Signal's Vector becomes a Prototype in the session domain.
// All prior turn weights are decayed — recent turns matter more.
//
// Call this after Engine.Run completes for each conversation turn.
func (s *Store) Record(signal core.Signal, turnIndex int) error {
	if len(signal.Vector) == 0 {
		return nil // nothing to record — empty signal
	}

	// Decay all existing entries.
	for i := range s.turns {
		id := fmt.Sprintf("turn.%d", s.turns[i].Index)
		entry, err := s.ks.Get(SessionDomain, id)
		if err != nil {
			continue
		}
		decayed := entry.Prototype.Weight * s.decay
		_ = s.ks.UpdateWeight(SessionDomain, id, decayed)
	}

	// Store current turn as a new Prototype at full weight.
	id := fmt.Sprintf("turn.%d", turnIndex)
	entry := knowledge.Entry{
		ID:        id,
		Prototype: core.NewPrototype(signal.Vector, 1.0),
		Surface:   "", // session Prototypes have no surface — they are for reasoning only
	}

	if err := s.ks.Load(SessionDomain, knowledge.Intelligence, []knowledge.Entry{entry}); err != nil {
		return fmt.Errorf("imnetia/session: record turn %d: %w", turnIndex, err)
	}

	s.turns = append(s.turns, Turn{Index: turnIndex, Signal: signal})
	return nil
}

// KnowledgeStore returns the session's KnowledgeStore.
// Pass this to the Engine alongside the permanent KnowledgeStore,
// or merge it into the Context so nodes can query SessionDomain.
//
// Nodes query session context via: ctx.Knowledge(session.SessionDomain)
func (s *Store) KnowledgeStore() *knowledge.KnowledgeStore {
	return s.ks
}

// Turns returns all recorded turns in order.
// Used by imnetia/learning to treat the session as a training episode.
func (s *Store) Turns() []Turn {
	out := make([]Turn, len(s.turns))
	copy(out, s.turns)
	return out
}

// Len returns the number of turns recorded so far.
func (s *Store) Len() int {
	return len(s.turns)
}

// Clear resets the session store — discards all recorded turns.
// Call this to start a new conversation with a clean slate.
func (s *Store) Clear() {
	s.ks = knowledge.New()
	s.turns = s.turns[:0]
}

// MostRecent returns the Signal from the most recently recorded turn,
// and true. Returns zero Signal and false if no turns have been recorded.
func (s *Store) MostRecent() (core.Signal, bool) {
	if len(s.turns) == 0 {
		return core.Signal{}, false
	}
	return s.turns[len(s.turns)-1].Signal, true
}
