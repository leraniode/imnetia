package runtime

import (
	"github.com/leraniode/imnetia/types"
	"github.com/leraniode/xgo/centrix/core"
)

// contextEngine constructs the initial Signal from a Context.
// Internal forever — its implementation evolves as encoding theory matures.
//
// v1: developer places an encoded SparseVector in Context under key "input".
// contextEngine wraps it into a Signal with Confidence 1.0.
// If absent, returns an empty Signal with Confidence 0.0.
type contextEngine struct{}

const inputKey = "input"

func newContextEngine() *contextEngine { return &contextEngine{} }

func (ce *contextEngine) Project(ctx types.Context) core.Signal {
	vec := ctx.Vector(inputKey)
	if vec == nil {
		return core.NewSignal(core.SparseVector{}, 0.0)
	}
	return core.NewSignal(vec, 1.0)
}
