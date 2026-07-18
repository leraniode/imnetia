package runtime

import (
	"github.com/leraniode/imnetia/encoding"
	"github.com/leraniode/imnetia/types"
	"github.com/leraniode/x/centrix/core"
	"github.com/leraniode/x/centrix/registry"
)

// contextEngine constructs the initial Signal from a Context.
// Internal forever — its implementation evolves as encoding theory matures.
//
// Two paths, checked in priority order:
//
//  1. "input" key — developer provides a pre-encoded SparseVector directly.
//     Used for testing, structured input, or when the developer wants
//     full control over the feature representation.
//
//  2. "query" key — developer provides raw text.
//     The contextEngine encodes it automatically using the Registry-based
//     tokeniser. This is the standard path for user-facing text input.
//
// If neither key is present, returns an empty Signal with Confidence 0.0.
// The flow still runs — nodes may build meaning from knowledge lookups.
type contextEngine struct {
	enc *encoding.Encoder
}

const (
	inputKey = "input" // pre-encoded SparseVector
	queryKey = "query" // raw text — encoded automatically
)

func newContextEngine(reg *registry.Registry) *contextEngine {
	return &contextEngine{
		enc: encoding.New(reg),
	}
}

func (ce *contextEngine) Project(ctx types.Context) core.Signal {
	// Priority 1: pre-encoded vector (developer-controlled).
	if vec := ctx.Vector(inputKey); vec != nil {
		return core.NewSignalFromVector(vec, 1.0)
	}

	// Priority 2: raw text query (auto-encoded).
	if query := ctx.String(queryKey); query != "" {
		vec := ce.enc.Encode(query)
		if len(vec) > 0 {
			return core.NewSignalFromVector(vec, 1.0)
		}
		// Query had no Registry hits — empty signal.
		// Honest: the system doesn't know any of those words.
	}

	// No input: empty signal, zero confidence.
	return core.NewSignal(0)
}
