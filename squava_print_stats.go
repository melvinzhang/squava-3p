//go:build !wasm

package main

import (
	"fmt"
)

type MoveStat struct {
	mv      Move
	visits  int
	winrate float32
}

func (m *MCTSPlayer) PrintStats(myID int, totalSteps, rollouts int) {
	if !m.Verbose {
		return
	}
	root := m.root
	fmt.Printf("Rollouts: %d, Steps: %d\n", rollouts, totalSteps)
	fmt.Printf("Estimated Winrate: %.2f%%\n", root.Q[myID]*100)

	stats := []MoveStat{}
	bestVisits := -1
	for i := range root.Edges {
		edge := &root.Edges[i]
		mv := edge.Move
		visits := int(edge.N)
		q := root.EdgeQs[i]
		stats = append(stats, MoveStat{mv, visits, q})
		if visits > bestVisits {
			bestVisits = visits
		}
	}
	// Sort stats by visits descending
	for i := 0; i < len(stats); i++ {
		for j := i + 1; j < len(stats); j++ {
			if stats[i].visits < stats[j].visits {
				stats[i], stats[j] = stats[j], stats[i]
			}
		}
	}
	fmt.Println("Top moves:")
	limit := 5
	if len(stats) < limit {
		limit = len(stats)
	}
	for i := 0; i < limit; i++ {
		s := stats[i]
		fmt.Printf("  %c%d: Visits: %d, Winrate: %.2f%%\n", int(s.mv.c)+65, int(s.mv.r)+1, s.visits, s.winrate*100)
	}
}
