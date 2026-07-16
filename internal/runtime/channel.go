// Package runtime is the internal execution engine of Imnetia.
// Nothing in this package is part of the public API.
// Developers never import or interact with it directly.
package runtime

import "github.com/leraniode/xgo/centrix/core"

// nodeChannel is a buffered Go channel that carries a Signal between
// two node goroutines in the execution pipeline.
//
// Buffered at 1 — the sender does not block waiting for the receiver
// to be ready, as long as the previous Signal has been consumed.
// This keeps the pipeline flowing without deadlock.
//
// Signal is a value type — copying through a channel is safe and correct.
// No pointer aliasing, no shared mutable state between goroutines.
type nodeChannel chan core.Signal

// newNodeChannel creates a buffered NodeChannel with capacity 1.
func newNodeChannel() nodeChannel {
	return make(chan core.Signal, 1)
}

// send puts a Signal into the channel.
// Non-blocking because buffer size is 1 and the executor guarantees
// the channel is empty before sending.
func (nc nodeChannel) send(s core.Signal) {
	nc <- s
}

// recv receives a Signal from the channel.
// Blocks until a Signal is available — the goroutine waits here
// until the upstream node has finished and sent its result.
func (nc nodeChannel) recv() core.Signal {
	return <-nc
}
