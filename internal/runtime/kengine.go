package runtime

import (
	"sort"

	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/x/centrix/core"
)

// knowledgeEngine performs Cosine-based similarity search against Prototypes.
// Internal only — developers access knowledge via ctx.Knowledge().
type knowledgeEngine struct {
	store *knowledge.KnowledgeStore
}

func newKnowledgeEngine(store *knowledge.KnowledgeStore) *knowledgeEngine {
	return &knowledgeEngine{store: store}
}

type scoredEntry struct {
	entry knowledge.Entry
	score float64
}

// Query returns entries from domain ranked by Cosine(query, prototype) × weight.
// Entries with zero or negative Cosine similarity are excluded.
// topK=0 returns all matching entries.
func (ke *knowledgeEngine) Query(query core.Signal, domain string, topK int) []knowledge.Entry {
	entries, err := ke.store.Domain(domain)
	if err != nil || len(entries) == 0 {
		return nil
	}

	scored := make([]scoredEntry, 0, len(entries))
	for _, e := range entries {
		sim := core.Cosine(query.Vector, e.Prototype.Vector)
		if sim <= 0 {
			continue
		}
		scored = append(scored, scoredEntry{entry: e, score: sim * e.Prototype.Weight})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	if topK > 0 && len(scored) > topK {
		scored = scored[:topK]
	}

	result := make([]knowledge.Entry, len(scored))
	for i, s := range scored {
		result[i] = s.entry
	}
	return result
}

// TopMatch returns the single highest-scoring Prototype and its score.
func (ke *knowledgeEngine) TopMatch(query core.Signal, domain string) (knowledge.Entry, float64) {
	entries, err := ke.store.Domain(domain)
	if err != nil || len(entries) == 0 {
		return knowledge.Entry{}, 0
	}

	var best knowledge.Entry
	var bestScore float64
	for _, e := range entries {
		sim := core.Cosine(query.Vector, e.Prototype.Vector)
		if sim <= 0 {
			continue
		}
		if s := sim * e.Prototype.Weight; s > bestScore {
			bestScore = s
			best = e
		}
	}
	return best, bestScore
}
