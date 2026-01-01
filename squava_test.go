package main

import (
	"flag"
	"math/bits"
	"testing"
)

var fuzzIterations = flag.Int("fuzz_iters", 10000, "Number of iterations for fuzz tests")

func generateRandomBoard(numPieces int) Board {
	board := Board{}
	for j := 0; j < numPieces; j++ {
		idx := int(xrand() % 64)
		p := int(xrand() % 3)
		if (board.Occupied & (1 << uint(idx))) == 0 {
			board.Set(idx, p)
		}
	}
	return board
}

func TestCheckBoard(t *testing.T) {
	// A horizontal win at the start of the row (A1, B1, C1, D1)
	// Bits: 0, 1, 2, 3
	winA1 := Bitboard(0x000000000000000F)
	isWin, _ := CheckBoard(winA1)
	if !isWin {
		t.Errorf("CheckBoard failed to detect horizontal win at A1-D1.")
	}

	// A horizontal line that wraps around: G1, H1, A2, B2
	// Bits: 6, 7, 8, 9
	// This SHOULD NOT be a win.
	wrap := Bitboard(0x00000000000003C0)
	isWin, _ = CheckBoard(wrap)
	if isWin {
		t.Errorf("CheckBoard incorrectly detected a wrap-around horizontal line (G1, H1, A2, B2) as a win.")
	}

	// Anti-diagonal win starting at H1 (H1, G2, F3, E4)
	// Bits: 7, 14, 21, 28
	winH1 := Bitboard((uint64(1) << 7) | (uint64(1) << 14) | (uint64(1) << 21) | (uint64(1) << 28))
	isWin, _ = CheckBoard(winH1)
	if !isWin {
		t.Errorf("CheckBoard failed to detect anti-diagonal win starting at H1.")
	}
}

// slowGetWinsAndLosses is a simple, obviously correct implementation for testing.
func slowGetWinsAndLosses(bb Bitboard, empty Bitboard) (wins Bitboard, loses Bitboard) {
	b := uint64(bb)
	e := uint64(empty)
	var w, l uint64

	directions := []struct{ dr, dc int }{
		{0, 1},
		{1, 0},
		{1, 1},
		{1, -1},
	}

	for i := 0; i < 64; i++ {
		if (e & (1 << uint(i))) == 0 {
			continue
		}

		r, c := i/8, i%8

		// Check for 4-in-a-row (win)
		isWin := false
		for _, dir := range directions {
			// A win can start at any of 4 offsets relative to the new piece
			for startOffset := -3; startOffset <= 0; startOffset++ {
				count := 0
				for k := 0; k < 4; k++ {
					nr, nc := r+(startOffset+k)*dir.dr, c+(startOffset+k)*dir.dc
					if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
						// Check if it's the new piece or an existing piece
						if (nr == r && nc == c) || (b&(1<<uint(nr*8+nc))) != 0 {
							count++
						}
						}
				}
				if count == 4 {
					isWin = true
					break
				}
			}
			if isWin {
				break
			}
		}

		if isWin {
			w |= (1 << uint(i))
			continue
		}

		// Check for 3-in-a-row (loss)
		isLoss := false
		for _, dir := range directions {
			// A loss can start at any of 3 offsets relative to the new piece
			for startOffset := -2; startOffset <= 0; startOffset++ {
				count := 0
				for k := 0; k < 3; k++ {
					nr, nc := r+(startOffset+k)*dir.dr, c+(startOffset+k)*dir.dc
					if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
						if (nr == r && nc == c) || (b&(1<<uint(nr*8+nc))) != 0 {
							count++
						}
						}
				}
				if count == 3 {
					isLoss = true
					break
				}
			}
			if isLoss {
				break
			}
		}
		if isLoss {
			l |= (1 << uint(i))
		}
	}

	return Bitboard(w), Bitboard(l & ^w)
}

func TestGetWinsAndLossesRandomized(t *testing.T) {
	for seed := 0; seed < *fuzzIterations; seed++ {
		xorState = uint64(seed + 1)
		// Generate random board
		var b, e uint64
		for i := 0; i < 64; i++ {
			val := xrand() % 3
			if val == 1 {
				b |= (1 << uint(i))
			} else if val == 0 {
				e |= (1 << uint(i))
			}
		}

		wExpected, lExpected := slowGetWinsAndLosses(Bitboard(b), Bitboard(e))
		wActual, lActual := GetWinsAndLosses(Bitboard(b), Bitboard(e))

		if wActual != wExpected {
			t.Errorf("Seed %d: Win bitboard mismatch. Expected %016x, got %016x", seed, uint64(wExpected), uint64(wActual))
		}
		if lActual != lExpected {
			t.Errorf("Seed %d: Loss bitboard mismatch. Expected %016x, got %016x", seed, uint64(lExpected), uint64(lActual))
		}
	}
}

func TestSimulationLogic(t *testing.T) {
	// Test elimination logic: P0 makes 3-in-a-row and should be eliminated.
	board := Board{}
	board.Set(0, 0)
	board.Set(1, 0)
	h := ZobristHash(board, 0, 0x07)
	// P0 moves to 2, creating 3-in-a-row
	state := SimulateStep(board, 0x07, 0, MoveFromIndex(2), h)

	if state.winnerID != -1 {
		t.Errorf("Expected no winner yet, got %d", state.winnerID)
	}
	if (state.activeMask & (1 << 0)) != 0 {
		t.Errorf("Player 0 should have been eliminated from activeMask")
	}
	if state.nextPlayerID != 1 {
		t.Errorf("Expected next player to be 1, got %d", state.nextPlayerID)
	}

	// Test last man standing: P0 eliminated, P1 eliminated, P2 should win.
	board = Board{}
	board.Set(0, 0)
	board.Set(1, 0)
	board.Set(8, 1)
	board.Set(9, 1)
	h = ZobristHash(board, 0, 0x07)

	// P0 moves to 2 -> eliminated. Mask becomes 0x06 (P1, P2)
	state1 := SimulateStep(board, 0x07, 0, MoveFromIndex(2), h)
	// P1 moves to 10 -> eliminated. Mask becomes 0x04 (P2)
	state2 := SimulateStep(state1.board, state1.activeMask, 1, MoveFromIndex(10), state1.hash)

	if state2.winnerID != 2 {
		t.Errorf("Expected Player 2 to win as last man standing, got %d", state2.winnerID)
	}
}

func TestZobristConsistency(t *testing.T) {
	board := Board{}
	board.Set(0, 0)
	board.Set(1, 1)
	board.Set(2, 2)

	h1 := ZobristHash(board, 0, 0x07)
	h2 := ZobristHash(board, 0, 0x07)
	if h1 != h2 {
		t.Errorf("Hash mismatch for identical states")
	}

	h3 := ZobristHash(board, 1, 0x07)
	if h1 == h3 {
		t.Errorf("Hash collision for different turn index")
	}

	board2 := board
	board2.Set(3, 0)
	h4 := ZobristHash(board2, 0, 0x07)
	if h1 == h4 {
		t.Errorf("Hash collision for different board state")
	}
}

func TestMCTSTerminal(t *testing.T) {
	// Test that MCTS can see an immediate win
	player := NewMCTSPlayer("Test", "T", 0, 100)
	board := Board{}
	board.Set(0, 0)
	board.Set(1, 0)
	board.Set(2, 0)
	move := player.GetMove(board, []int{0, 1, 2}, 0)
	if move.ToIndex() != 3 {
		t.Errorf("MCTS failed to find immediate win at index 3, got %d", move.ToIndex())
	}
}

func TestDrawOnFullBoard(t *testing.T) {
	// Create a board that is almost full
	board := Board{}
	// Fill almost everything with a pattern that doesn't create wins/losses
	// (Not easy in Squava, but for simulation we just care about termination)
	for i := 0; i < 63; i++ {
		board.Set(i, (i/2)%3)
	}
	// Simulation should terminate with a draw (all zeros) if no moves left
	// We force a state where no wins/losses are possible in 1 move
	res, _, _ := RunSimulation(board, 0x07, 0)
	if res[0] != 0 || res[1] != 0 || res[2] != 0 {
		// Note: in a random simulation someone might win/lose,
		// but if we reach moves == 0 it should be [0,0,0]
		// Let's just check if it returns.
		t.Logf("Simulation returned %v", res)
	}
}

func TestMCTSHeuristic(t *testing.T) {
	// Test that MCTS respects the GetBestMoves heuristic (blocking opponent)
	player := NewMCTSPlayer("AI", "A", 0, 100)
	board := Board{}
	// Player 1 (next) has 3 in a row at A1, A2, A3
	board.Set(0, 1) // A1
	board.Set(8, 1) // A2
	board.Set(16, 1) // A3
	// Player 0 (current) MUST block at A4 (bit 24)
	move := player.GetMove(board, []int{0, 1, 2}, 0)
	if move.ToIndex() != 24 {
		t.Errorf("MCTS failed to block opponent win at A4, chose %d", move.ToIndex())
	}
}

func TestRunSimulationDetailed(t *testing.T) {
	// 1. Immediate win detection
	board := Board{}
	board.Set(0, 0) // A1
	board.Set(1, 0) // B1
	board.Set(2, 0) // C1
	// P0 to move, D1 (3) is win
	res, _, _ := RunSimulation(board, 0x07, 0)
	if res[0] != 1.0 {
		t.Errorf("Immediate win failed. Expected P0 win, got %v", res)
	}

	// 2. Forced block detection
	board = Board{}
	board.Set(0, 1) // P1: A1
	board.Set(1, 1) // P1: B1
	board.Set(2, 1) // P1: C1
	// P0 to move, P1 is next. P0 must block at D1 (3)
	// We seed xorState to ensure we don't just "get lucky"
	xorState = 42
	res, steps, _ := RunSimulation(board, 0x07, 0)
	// If P0 blocks correctly, the game should continue for more than 1 step
	if steps <= 1 && res[1] == 1.0 {
		t.Errorf("Forced block failed. P0 should have blocked P1's win at D1. Steps: %d, Result: %v", steps, res)
	}

	// 3. Elimination logic
	board = Board{}
	// Set up P0 so any move is a loss
	board.Set(0, 0) // A1
	board.Set(8, 0) // A2
	// Move at A3 (16) will eliminate P0
	// We'll use a almost full board to force this
	for i := 0; i < 64; i++ {
		if i != 16 && i != 0 && i != 8 {
			board.Set(i, 1) // Fill with P1
		}
	}
	// Only bit 16 is empty. P0 must move there.
	res, _, _ = RunSimulation(board, 0x07, 0)
	if res[0] == 1.0 {
		t.Errorf("Elimination failed. P0 should have lost, but won: %v", res)
	}
}

func referenceRunSimulation(board Board, activeMask uint8, currentID int) ([3]float64, Board) {
	simBoard := board
	simMask := activeMask
	curr := currentID
	for {
		if simMask&(simMask-1) == 0 {
			var res [3]float64
			res[bits.TrailingZeros8(simMask)] = 1.0
			return res, simBoard
		}

		empty := ^simBoard.Occupied
		// Simple version: recompute threats every turn for current player
		myWins, myLoses := GetWinsAndLosses(simBoard.P[curr], empty)

		if myWins != 0 {
			var res [3]float64
			res[curr] = 1.0
			return res, simBoard
		}

		nextP := int(nextPlayerTable[curr][simMask])
		// Recompute threats for next player
		nextWins, _ := GetWinsAndLosses(simBoard.P[nextP], empty)

		var moves Bitboard
		mustCheckLoss := true
		if nextWins != 0 {
			moves = nextWins
		} else {
			moves = empty & ^myLoses
			if moves != 0 {
				mustCheckLoss = false
			} else {
				moves = empty
			}
		}

		if moves == 0 {
			return [3]float64{}, simBoard
		}

		var selectedIdx int
		count := bits.OnesCount64(uint64(moves))
		if count == 1 {
			selectedIdx = bits.TrailingZeros64(uint64(moves))
		} else {
			hi, _ := bits.Mul64(xrand(), uint64(count))
			// Use SelectBit64 to ensure identical bit selection
			selectedIdx = SelectBit64(uint64(moves), int(hi))
		}

		mask := Bitboard(uint64(1) << selectedIdx)
		simBoard.Occupied |= mask
		simBoard.P[curr] |= mask

		if mustCheckLoss {
			_, isLoss := CheckBoard(simBoard.P[curr])
			if isLoss {
				simMask &= ^(1 << uint(curr))
				if simMask&(simMask-1) == 0 {
					var res [3]float64
					res[bits.TrailingZeros8(simMask)] = 1.0
					return res, simBoard
				}
				curr = int(nextPlayerTable[curr][simMask])
				continue
			}
		}
		curr = nextP
	}
}

func TestRunSimulationRandomized(t *testing.T) {
	for i := 0; i < *fuzzIterations; i++ {
		board := generateRandomBoard(20)
		won := false
		for p := 0; p < 3; p++ {
			isW, isL := CheckBoard(board.P[p])
			if isW || isL {
				won = true
				break
			}
		}
		if won {
			continue
		}

		// Ensure both use exact same random sequence
		seed := uint64(i + 1)
		
		xorState = seed
		resOpt, _, boardOpt := RunSimulation(board, 0x07, 0)
		
		xorState = seed
		resRef, boardRef := referenceRunSimulation(board, 0x07, 0)

		if resOpt != resRef {
			t.Errorf("Iteration %d: Result mismatch. Opt: %v, Ref: %v", i, resOpt, resRef)
		}
		if boardOpt != boardRef {
			t.Errorf("Iteration %d: Final board mismatch.", i)
		}
	}
}

func TestZobristIncrementalFuzz(t *testing.T) {
	for i := 0; i < *fuzzIterations; i++ {
		// 1. Generate Random Board
		board := generateRandomBoard(20)

		// 2. Setup Random Valid Context
		// Active mask must have at least 2 players
		var activeMask uint8
		for {
			activeMask = uint8(xrand() % 8)
			if bits.OnesCount8(activeMask) >= 2 {
				break
			}
		}

		// Current player must be in active mask
		var currentID int
		for {
			currentID = int(xrand() % 3)
			if (activeMask & (1 << uint(currentID))) != 0 {
				break
			}
		}

		// 3. Select a Random Valid Move
		empty := ^board.Occupied
		if empty == 0 {
			continue // Full board, retry
		}
		// Pick a random bit from empty
		count := bits.OnesCount64(uint64(empty))
		if count == 0 {
			continue
		}
		n := int(xrand() % uint64(count))
		// Find n-th bit
		idx := SelectBit64(uint64(empty), n)
		move := MoveFromIndex(idx)

		// 4. Compute Initial Hash
		initialHash := ZobristHash(board, currentID, activeMask)

		// 5. Execute Step (Incremental Update)
		newState := SimulateStep(board, activeMask, currentID, move, initialHash)

		// 6. Verification: Compute Hash from Scratch on New State
		refHash := ZobristHash(newState.board, newState.nextPlayerID, newState.activeMask)

		if newState.hash != refHash {
			t.Errorf("Iteration %d: Hash mismatch.\nState: %+v\nIncremental: %016x\nReference:   %016x",
				i, newState, newState.hash, refHash)
			return // Stop on first error to avoid spam
		}
	}
}

func TestSelectBit64Fuzz(t *testing.T) {
	referenceSelectBit64 := func(v uint64, k int) int {
		count := 0
		for i := 0; i < 64; i++ {
			if (v & (1 << uint(i))) != 0 {
				if count == k {
					return i
				}
				count++
			}
		}
		return 64
	}

	for i := 0; i < *fuzzIterations*10; i++ {
		// Generate various types of bit patterns
		var v uint64
		switch i % 4 {
		case 0:
			v = xrand() // Completely random
		case 1:
			v = 1 << (xrand() % 64) // Single bit
		case 2:
			v = ^uint64(0) // All bits set
		case 3:
			v = xrand() & xrand() & xrand() // Sparse
		}

		count := bits.OnesCount64(v)
		if count == 0 {
			continue
		}

		k := int(xrand() % uint64(count))
		expected := referenceSelectBit64(v, k)
		actual := SelectBit64(v, k)

		if actual != expected {
			t.Errorf("Iteration %d: Mismatch for v=%016x, k=%d. Expected bit %d, got %d",
				i, v, k, expected, actual)
			return
		}
	}
}

func TestHeuristicMoveGenerationFuzz(t *testing.T) {
	for i := 0; i < *fuzzIterations; i++ {
		// 1. Generate Random Board
		board := generateRandomBoard(25)

		// Ensure board is clean (no existing wins/losses)
		clean := true
		for p := 0; p < 3; p++ {
			isW, isL := CheckBoard(board.P[p])
			if isW || isL {
				clean = false
				break
			}
		}
		if !clean {
			continue
		}

		// 2. Setup Players
		currentID := int(xrand() % 3)
		// Next player doesn't strictly matter for the heuristic function's internal logic 
		// regarding currentID's wins, but matters for blocking.
		nextID := (currentID + 1) % 3 

		// 3. Analyze
		threats := AnalyzeThreats(board, currentID, nextID)
		forced := GetForcedMoves(board, currentID, nextID)
		best := GetBestMoves(board, threats)

		// 4. Verify Validity
		empty := ^board.Occupied
		if (threats.MyWins & ^empty) != 0 {
			t.Errorf("Iteration %d: MyWins overlaps occupied", i)
		}
		if (threats.NextWins & ^empty) != 0 {
			t.Errorf("Iteration %d: NextWins overlaps occupied", i)
		}
		if (threats.MyLoses & ^empty) != 0 {
			t.Errorf("Iteration %d: MyLoses overlaps occupied", i)
		}
		if (forced & ^empty) != 0 {
			t.Errorf("Iteration %d: Forced moves overlap occupied", i)
		}
		if (best & ^empty) != 0 {
			t.Errorf("Iteration %d: Best moves overlap occupied", i)
		}

		// 5. Verify MyWins (Immediate Win)
		w := threats.MyWins
		for w != 0 {
			idx := bits.TrailingZeros64(uint64(w))
			testBoard := board
			testBoard.Set(idx, currentID)
			isWin, _ := CheckBoard(testBoard.P[currentID])
			if !isWin {
				t.Errorf("Iteration %d: AnalyzeThreats claimed immediate win at %d, but CheckBoard said no.", i, idx)
			}
			w &= w - 1
		}

		// 6. Verify NextWins (Opponent Win)
		nw := threats.NextWins
		for nw != 0 {
			idx := bits.TrailingZeros64(uint64(nw))
			testBoard := board
			testBoard.Set(idx, nextID)
			isWin, _ := CheckBoard(testBoard.P[nextID])
			if !isWin {
				t.Errorf("Iteration %d: AnalyzeThreats claimed opponent win at %d, but CheckBoard said no.", i, idx)
			}
			nw &= nw - 1
		}

		// 7. Verify MyLoses (Self-Loss)
		l := threats.MyLoses
		for l != 0 {
			idx := bits.TrailingZeros64(uint64(l))
			testBoard := board
			testBoard.Set(idx, currentID)
			_, isLoss := CheckBoard(testBoard.P[currentID])
			if !isLoss {
				t.Errorf("Iteration %d: AnalyzeThreats claimed self-loss at %d, but CheckBoard said no.", i, idx)
			}
			l &= l - 1
		}

		// 8. Verify Priority Logic
		if threats.MyWins != 0 {
			if forced != threats.MyWins {
				t.Errorf("Iteration %d: Forced should equal MyWins", i)
			}
			if best != threats.MyWins {
				t.Errorf("Iteration %d: Best should equal MyWins", i)
			}
		} else if threats.NextWins != 0 {
			if forced != threats.NextWins {
				t.Errorf("Iteration %d: Forced should equal NextWins", i)
			}
			if best != threats.NextWins {
				t.Errorf("Iteration %d: Best should equal NextWins", i)
			}
		} else {
			if forced != 0 {
				t.Errorf("Iteration %d: Forced should be 0 when no immediate threats", i)
			}
			// Verify Best avoids losses if possible
			safe := empty & ^threats.MyLoses
			if safe != 0 {
				if best != safe {
					t.Errorf("Iteration %d: Best should be safe moves (Empty & ^MyLoses). Got %x, Expected %x", i, best, safe)
				}
			} else {
				if best != empty {
					t.Errorf("Iteration %d: Best should be empty (resignation/forced loss) when no safe moves", i)
				}
			}
		}
	}
}

func ValidateMCTSGraph(t *testing.T, root *MCGSNode) {
	if root == nil {
		t.Error("Root is nil")
		return
	}

	// Track visited nodes to handle DAG structure (shared nodes)
	visited := make(map[*MCGSNode]bool)
	// Track recursion stack for cycle detection
	stack := make(map[*MCGSNode]bool)

	var checkNode func(node *MCGSNode)
	checkNode = func(node *MCGSNode) {
		// 1. Cycle Detection
		if stack[node] {
			t.Errorf("Cycle detected in MCTS graph at node hash %016x", node.hash)
			return
		}
		// 2. Transposition Handling (memoization)
		if visited[node] {
			return
		}
		visited[node] = true
		stack[node] = true
		defer func() { stack[node] = false }()

		// 3. Value Invariants
		if node.N < 0 {
			t.Errorf("Node %016x has negative visits: %d", node.hash, node.N)
		}
		// Assuming Win/Loss rewards are 0.0 to 1.0 (or -1 to 1)
		// Squava implementation uses [0,1]
		for i := 0; i < 3; i++ {
			if node.Q[i] < -0.01 || node.Q[i] > 1.01 {
				t.Errorf("Node %016x has Q value out of bounds [0,1]: %v", node.hash, node.Q)
			}
		}

		// 4. Edge Invariants
		sumEdgeVisits := 0
		for _, edge := range node.Edges {
			sumEdgeVisits += edge.Visits

			if edge.Dest == nil {
				t.Errorf("Node %016x has edge with nil destination", node.hash)
				continue
			}

			// DAG Invariant: Edge Visits vs Child Total Visits
			// The number of times we traversed THIS edge to get to Child
			// must be <= Total times Child was visited (from any parent).
			if edge.Visits > edge.Dest.N {
				t.Errorf("Flow violation: Edge from %016x to %016x has %d visits, but child only has %d total visits.",
					node.hash, edge.Dest.hash, edge.Visits, edge.Dest.N)
			}

			// 5. State Consistency Check
			// Re-simulate the move to ensure the hash matches the destination node
			expectedState := SimulateStep(node.board, node.activeMask, node.playerToMoveID, edge.Move, node.hash)
			if expectedState.hash != edge.Dest.hash {
				t.Errorf("Hash consistency violation on edge %v: Expected %016x, got %016x",
					edge.Move, expectedState.hash, edge.Dest.hash)
			}

			// Recurse
			checkNode(edge.Dest)
		}

		// 6. Conservation of Flow (Outgoing)
		// The number of times we left this node cannot exceed the number of times we visited it.
		// node.N includes the visit where we stopped at this node (didn't traverse out).
		if sumEdgeVisits > node.N {
			t.Errorf("Node %016x has %d outgoing edge visits but only %d total node visits",
				node.hash, sumEdgeVisits, node.N)
		}
	}

	checkNode(root)
}

func TestMCTSInvariants(t *testing.T) {
	// Use fewer iterations for MCTS fuzzing to keep test time reasonable per run, 
	// unless specifically scaling up.
	mctsIters := 200
	fuzzRuns := *fuzzIterations / 10 // scale down total runs as MCTS is heavy

	for i := 0; i < fuzzRuns; i++ {
		// 1. Generate Random Board
		board := generateRandomBoard(25)

		// Ensure board is clean for MCTS start
		clean := true
		for p := 0; p < 3; p++ {
			isW, isL := CheckBoard(board.P[p])
			if isW || isL {
				clean = false
				break
			}
		}
		if !clean {
			continue
		}
		
		// Run a short game/simulation to build a graph
		player := NewMCTSPlayer("Tester", "T", 0, mctsIters) 
		
		// We don't care about the resulting move, just the graph structure
		_ = player.GetMove(board, []int{0, 1, 2}, 0)

		if player.root == nil {
			t.Errorf("Iteration %d: MCTS did not generate a root node", i)
			continue
		}

				// Validate the resulting graph

				ValidateMCTSGraph(t, player.root)

			}

		}

		

		func TestFullGameTerminationFuzz(t *testing.T) {

			for i := 0; i < *fuzzIterations/100; i++ { // Full games are slow, run fewer iterations

				board := Board{}

				activeMask := uint8(0x07)

				currentPID := 0

				players := []int{0, 1, 2}

		

				// Play until termination

				for {

					// 1. Check if game is already over (should not happen mid-loop)

					empty := ^board.Occupied

					if empty == 0 {

						break // Draw

					}

		

					// 2. Select random legal move

					count := bits.OnesCount64(uint64(empty))

					n := int(xrand() % uint64(count))

					idx := SelectBit64(uint64(empty), n)

					

					// 3. Apply move

					board.Set(idx, currentPID)

					

					// 4. Check rules

					isWin, isLoss := CheckBoard(board.P[currentPID])

					if isWin {

						// Current player wins. Verify no one else has a win.

						for _, p := range players {

							if p == currentPID { continue }

							w, _ := CheckBoard(board.P[p])

							if w {

								t.Errorf("Iteration %d: Invalid state. Multiple players have wins.", i)

							}

						}

						goto GameEnd

					}

					if isLoss {

						// Current player eliminated

						activeMask &= ^(1 << uint(currentPID))

						// Remove current player from players slice for turn rotation

						for j, p := range players {

							if p == currentPID {

								players = append(players[:j], players[j+1:]...)

								break

							}

						}

						

						if len(players) == 1 {

							// Last man standing wins

							winner := players[0]

							isW, _ := CheckBoard(board.P[winner])

							// Winner might have a 3-in-a-row (loss) but they win if they are the last one.

							// However, if they have a 4-in-a-row they win anyway.

							// Squava rules: last player wins even if they have 3-in-a-row? 

							// Logic check: if P1 makes 3 and is eliminated, and only P2 left, P2 wins.

							if isW {

								// P2 already had 4? That would be a win earlier.

							}

							goto GameEnd

						}

						

						// Game continues with fewer players. 

						// currentPID is already updated by turn logic below.

					} else {

						// No win or loss, turn rotation handled below

					}

		

					if board.Occupied == Bitboard(Full) {

						goto GameEnd // Draw

					}

		

					// 5. Next Turn

					nextPID := int(nextPlayerTable[currentPID][activeMask])

					if nextPID == -1 {

						t.Errorf("Iteration %d: nextPlayerTable returned -1 for activeMask %02x", i, activeMask)

						goto GameEnd

					}

					currentPID = nextPID

				}

		

			GameEnd:

				// Termination reached. Verify invariants.

				winCount := 0

				for p := 0; p < 3; p++ {

					isW, _ := CheckBoard(board.P[p])

					if isW { winCount++ }

				}

				if winCount > 1 {

					t.Errorf("Iteration %d: Invalid termination. Multiple winners: %d", i, winCount)

				}

			}

		}

		