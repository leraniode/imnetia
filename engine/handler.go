// Package engine is the public execution surface of Imnetia.
// Developers import engine to run flows and access the Engine type.
// For writing NodeHandlers, import imnetia/types directly.
package engine

// This file re-exports types from imnetia/types for convenience,
// so developers only need to import imnetia/engine for the full surface.

import "github.com/leraniode/imnetia/types"

// NodeHandler is a function that transforms a Signal inside a Node.
// See types.NodeHandler for the full definition.
type NodeHandler = types.NodeHandler

// NodeResult carries routing information from a NodeHandler.
// See types.NodeResult for the full definition.
type NodeResult = types.NodeResult

// Context is the run-scoped operational store passed to each NodeHandler.
// See types.Context for the full definition.
type Context = types.Context

// NewContext creates a Context backed by the given KnowledgeStore.
// This is a convenience re-export of types.NewContext.
var NewContext = types.NewContext
