package engine_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/leraniode/imnetia/decode"
	"github.com/leraniode/imnetia/engine"
	"github.com/leraniode/imnetia/graph"
	"github.com/leraniode/imnetia/knowledge"
	"github.com/leraniode/imnetia/types"
	"github.com/leraniode/x/centrix/core"
	"github.com/leraniode/x/centrix/registry"
)

// goErrorsSystem builds the smallest complete system: a Registry whose concepts
// have namespaced names and surface-form aliases, a knowledge domain whose
// Prototypes use those same feature indexes, and a one-node flow.
func goErrorsSystem(t *testing.T) (*engine.Engine, *graph.Flow, *knowledge.KnowledgeStore) {
	t.Helper()

	reg := registry.New()
	for _, c := range []string{"lang.go", "concept.error", "concept.handle", "concept.file", "concept.read"} {
		reg.ID(c)
	}
	if err := reg.AliasLastSegments("lang.go", "concept.error", "concept.handle", "concept.file", "concept.read"); err != nil {
		t.Fatal(err)
	}
	// Plural/inflected forms are covered by the stemmer; an extra synonym is an alias.
	if err := reg.Alias("failure", "concept.error"); err != nil {
		t.Fatal(err)
	}

	ks := knowledge.New()
	err := ks.Load("go.errors", knowledge.Foundation, []knowledge.Entry{
		{
			ID: "go.errors.check",
			Prototype: core.NewPrototype(core.SparseVector{
				reg.ID("lang.go"): 1, reg.ID("concept.error"): 1, reg.ID("concept.handle"): 1,
			}, 0.9),
			Surface: "check errors with: if err != nil { return err }",
		},
		{
			ID: "go.files.read",
			Prototype: core.NewPrototype(core.SparseVector{
				reg.ID("lang.go"): 1, reg.ID("concept.file"): 1, reg.ID("concept.read"): 1,
			}, 0.9),
			Surface: "read a file with os.ReadFile",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	pass := func(ctx types.Context, in core.Signal) (core.Signal, types.NodeResult) {
		return in, types.NodeResult{}
	}
	f := graph.NewFlow()
	n, _ := graph.NewNode("understand", types.NodeHandler(pass))
	_ = f.Add(n)
	_ = f.Entry("understand")

	return engine.New(ks, reg), f, ks
}

// The registry-vocabulary-mismatch bug: namespaced concepts + raw text used to
// give an empty vector, Confidence 0.00 and "I don't know" for every query.
func TestQuery_RawTextReachesNamespacedConcepts(t *testing.T) {
	e, flow, ks := goErrorsSystem(t)

	ctx := types.NewContext(ks, e.Registry())
	ctx.Set("query", "how do I handle errors in go")

	sig, err := e.Run(flow, ctx)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(sig.Vector) == 0 || sig.Confidence <= 0 {
		t.Fatalf("query produced an empty signal (confidence %v, %d features)", sig.Confidence, len(sig.Vector))
	}

	resp, err := decode.Assemble(sig, ks, "go.errors", decode.DefaultThreshold, 0.3)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(resp.Fragments) == 0 || resp.Fragments[0] != "check errors with: if err != nil { return err }" {
		t.Fatalf("expected the error-handling fragment first, got %v", resp.Fragments)
	}
	if !resp.Result.Certain {
		t.Errorf("expected a certain result, got confidence %v", resp.Result.Confidence)
	}
}

func TestQuery_SynonymAliasAndOtherIntent(t *testing.T) {
	e, flow, ks := goErrorsSystem(t)

	run := func(q string) decode.AssembledResponse {
		ctx := types.NewContext(ks, e.Registry())
		ctx.Set("query", q)
		sig, err := e.Run(flow, ctx)
		if err != nil {
			t.Fatalf("Run(%q): %v", q, err)
		}
		resp, err := decode.Assemble(sig, ks, "go.errors", decode.DefaultThreshold, 0.3)
		if err != nil {
			t.Fatalf("Assemble(%q): %v", q, err)
		}
		return resp
	}

	// "failure" is an alias for concept.error.
	if got := run("go failure handle"); len(got.Fragments) == 0 {
		t.Error("alias 'failure' should reach concept.error")
	}
	// A query with no known words must stay honestly empty.
	if got := run("xyzzy florp quux"); len(got.Fragments) != 0 {
		t.Errorf("unknown words must not produce fragments, got %v", got.Fragments)
	}
}

// Same query, same system: bit-identical confidence and identical fragments.
func TestQuery_Deterministic(t *testing.T) {
	e, flow, ks := goErrorsSystem(t)

	type outcome struct {
		conf      uint64
		fragments []string
	}
	run := func() outcome {
		ctx := types.NewContext(ks, e.Registry())
		ctx.Set("query", "how do I handle errors reading a file in go")
		sig, err := e.Run(flow, ctx)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := decode.Assemble(sig, ks, "go.errors", decode.DefaultThreshold, 0.1)
		if err != nil {
			t.Fatal(err)
		}
		return outcome{math.Float64bits(sig.Confidence), resp.Fragments}
	}

	first := run()
	if len(first.fragments) < 2 {
		t.Fatalf("test needs both fragments to exercise ordering, got %v", first.fragments)
	}
	for i := 0; i < 100; i++ {
		if got := run(); !reflect.DeepEqual(got, first) {
			t.Fatalf("run %d differs:\n%+v\n%+v", i, first, got)
		}
	}
}
