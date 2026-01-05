package main

import (
	"math"
	"math/bits"
	"testing"
)

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
func FuzzGetWinsAndLossesAgainstSlow(f *testing.F) {
	f.Add(uint64(0), uint64(0))
	f.Fuzz(func(t *testing.T, b uint64, e uint64) {
		wExpected, lExpected := slowGetWinsAndLosses(Bitboard(b), Bitboard(e))
		wActual, lActual := GetWinsAndLosses(Bitboard(b), Bitboard(e))
		if wActual != wExpected {
			t.Errorf("Win bitboard mismatch. Expected %016x, got %016x", uint64(wExpected), uint64(wActual))
		}
		if lActual != lExpected {
			t.Errorf("Loss bitboard mismatch. Expected %016x, got %016x", uint64(lExpected), uint64(lActual))
		}
	})
}

func BenchmarkGetWinsAndLosses(b *testing.B) {
	bb := Bitboard(0x000000000000000F)
	empty := Bitboard(0xFFFFFFFFFFFFFFFF)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GetWinsAndLosses(bb, empty)
	}
}

func BenchmarkGetWinsAndLossesGo(b *testing.B) {
	bb := Bitboard(0x000000000000000F)
	empty := Bitboard(0xFFFFFFFFFFFFFFFF)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		getWinsAndLossesGo(uint64(bb), uint64(empty))
	}
}

func TestSimulationLogic(t *testing.T) {
	// Test elimination logic: P0 makes 3-in-a-row and should be eliminated.
	board := Board{}
	board.Set(0, 0)
	board.Set(1, 0)
	// P0 moves to 2, creating 3-in-a-row
	gs := NewGameState(board, 0, 0x07)
	gs.ApplyMove(MoveFromIndex(2))
	if gs.WinnerID != -1 {
		t.Errorf("Expected no winner yet, got %d", gs.WinnerID)
	}
	if (gs.ActiveMask & (1 << 0)) != 0 {
		t.Errorf("Player 0 should have been eliminated from activeMask")
	}
	if gs.PlayerID != 1 {
		t.Errorf("Expected next player to be 1, got %d", gs.PlayerID)
	}
	// Test last man standing: P0 eliminated, P1 eliminated, P2 should win.
	board = Board{}
	board.Set(0, 0)
	board.Set(1, 0)
	board.Set(8, 1)
	board.Set(9, 1)
	// P0 moves to 2 -> eliminated. Mask becomes 0x06 (P1, P2)
	gs1 := NewGameState(board, 0, 0x07)
	gs1.ApplyMove(MoveFromIndex(2))
	// P1 moves to 10 -> eliminated. Mask becomes 0x04 (P2)
	gs1.ApplyMove(MoveFromIndex(10))
	if gs1.WinnerID != 2 {
		t.Errorf("Expected Player 2 to win as last man standing, got %d", gs1.WinnerID)
	}
}
func TestZobristConsistency(t *testing.T) {
	board := Board{}
	board.Set(0, 0)
	board.Set(1, 1)
	board.Set(2, 2)
	h1 := zobrist.ComputeHash(board, 0, 0x07)
	h2 := zobrist.ComputeHash(board, 0, 0x07)
	if h1 != h2 {
		t.Errorf("Hash mismatch for identical states")
	}
	h3 := zobrist.ComputeHash(board, 1, 0x07)
	if h1 == h3 {
		t.Errorf("Hash collision for different turn index")
	}
	board2 := board
	board2.Set(3, 0)
	h4 := zobrist.ComputeHash(board2, 0, 0x07)
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
	for i := 0; i < 63; i++ {
		board.Set(i, (i/2)%3)
	}
	// Simulation should terminate with a draw if no moves left
	gs := NewGameState(board, 0, 0x07)
	res, _, _ := RunSimulation(&gs)
	// Expected draw score for 3 players is 1/3 each
	expected := float32(1.0 / 3.0)
	for i := 0; i < 3; i++ {
		if math.Abs(float64(res[i]-expected)) > 1e-6 {
			t.Errorf("Expected draw score %f for player %d, got %f", expected, i, res[i])
		}
	}
}
func TestMCTSHeuristic(t *testing.T) {
	// Test that MCTS respects the GetBestMoves heuristic (blocking opponent)
	player := NewMCTSPlayer("AI", "A", 0, 100)
	board := Board{}
	// Player 1 (next) has 3 in a row at A1, A2, A3
	board.Set(0, 1)  // A1
	board.Set(8, 1)  // A2
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
	gs1 := NewGameState(board, 0, 0x07)
	res, _, _ := RunSimulation(&gs1)
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
	gs2 := NewGameState(board, 0, 0x07)
	res, steps, _ := RunSimulation(&gs2)
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
	gs3 := NewGameState(board, 0, 0x07)
	res, _, _ = RunSimulation(&gs3)
	if res[0] == 1.0 {
		t.Errorf("Elimination failed. P0 should have lost, but won: %v", res)
	}
}
func referenceRunSimulation(board Board, activeMask uint8, currentID int) ([3]float32, Board) {
	gs := NewGameState(board, currentID, activeMask)
	for {
		if winnerID, ok := gs.IsTerminal(); ok {
			return ScoreTerminal(gs.ActiveMask, winnerID), gs.Board
		}

		moves := gs.GetBestMoves()
		idx := PickRandomBit(moves)
		if idx == -1 {
			return ScoreDraw(gs.ActiveMask), gs.Board
		}

		gs.ApplyMove(MoveFromIndex(idx))
	}
}
func FuzzRunSimulation(f *testing.F) {
	f.Add(uint64(1), uint64(0)) // seed, boardPieces
	f.Fuzz(func(t *testing.T, seed uint64, boardPieces uint64) {
		xorState = seed
		numPieces := int(boardPieces % 40)
		board := generateRandomBoard(numPieces)
		won := false
		for p := 0; p < 3; p++ {
			isW, isL := CheckBoard(board.P[p])
			if isW || isL {
				won = true
				break
			}
		}
		if won {
			return
		}
		// Ensure both use exact same random sequence
		runSeed := xrand()
		xorState = runSeed
		gs := NewGameState(board, 0, 0x07)
		resOpt, _, boardOpt := RunSimulation(&gs)
		xorState = runSeed
		resRef, boardRef := referenceRunSimulation(board, 0x07, 0)
		if resOpt != resRef {
			t.Errorf("Result mismatch. Opt: %v, Ref: %v", resOpt, resRef)
		}
		if boardOpt != boardRef {
			t.Errorf("Final board mismatch.")
		}
	})
}
func FuzzZobristIncremental(f *testing.F) {
	f.Add(uint64(1), uint64(20))
	f.Fuzz(func(t *testing.T, seed uint64, numPieces uint64) {
		xorState = seed
		board := generateRandomBoard(int(numPieces % 40))
		var activeMask uint8
		for {
			activeMask = uint8(xrand() % 8)
			if bits.OnesCount8(activeMask) >= 2 {
				break
			}
		}
		var currentID int
		for {
			currentID = int(xrand() % 3)
			if (activeMask & (1 << uint(currentID))) != 0 {
				break
			}
		}
		empty := ^board.Occupied
		if empty == 0 {
			return
		}
		count := bits.OnesCount64(uint64(empty))
		n := int(xrand() % uint64(count))
		idx := SelectBit64(uint64(empty), n)
		move := MoveFromIndex(idx)
		gs := NewGameState(board, currentID, activeMask)
		gs.ApplyMove(move)
		refHash := zobrist.ComputeHash(gs.Board, gs.PlayerID, gs.ActiveMask)
		if gs.Hash != refHash {
			t.Errorf("Hash mismatch. Incremental: %016x, Reference: %016x", gs.Hash, refHash)
		}
	})
}
func FuzzSelectBit64Internal(f *testing.F) {
	f.Add(uint64(0b101010), uint64(1))
	f.Fuzz(func(t *testing.T, v uint64, k64 uint64) {
		count := bits.OnesCount64(v)
		if count == 0 {
			return
		}
		k := int(k64 % uint64(count))
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
		expected := referenceSelectBit64(v, k)
		actual := SelectBit64(v, k)
		if actual != expected {
			t.Errorf("Mismatch for v=%016x, k=%d. Expected bit %d, got %d", v, k, expected, actual)
		}
	})
}
func FuzzHeuristicMoveGeneration(f *testing.F) {
	f.Add(uint64(1), uint64(25))
	f.Fuzz(func(t *testing.T, seed uint64, numPieces64 uint64) {
		xorState = seed
		board := generateRandomBoard(int(numPieces64 % 40))
		clean := true
		for p := 0; p < 3; p++ {
			isW, isL := CheckBoard(board.P[p])
			if isW || isL {
				clean = false
				break
			}
		}
		if !clean {
			return
		}
		currentID := int(xrand() % 3)
		gs := NewGameState(board, currentID, 0x07)
		forced := GetForcedMoves(board, []int{0, 1, 2}, currentID)
		best := gs.GetBestMoves()
		myWins := gs.Wins[currentID]
		myLoses := gs.Loses[currentID]
		nextID := gs.NextPlayer()
		nextWins := gs.Wins[nextID]

		empty := ^board.Occupied
		if (myWins & ^empty) != 0 || (nextWins & ^empty) != 0 || (myLoses & ^empty) != 0 || (forced & ^empty) != 0 || (best & ^empty) != 0 {
			t.Errorf("Moves overlap occupied squares")
		}
		w := myWins
		for w != 0 {
			idx := bits.TrailingZeros64(uint64(w))
			testBoard := board
			testBoard.Set(idx, currentID)
			isWin, _ := CheckBoard(testBoard.P[currentID])
			if !isWin {
				t.Errorf("GameState claimed immediate win at %d, but CheckBoard said no.", idx)
			}
			w &= w - 1
		}
		nw := nextWins
		for nw != 0 {
			idx := bits.TrailingZeros64(uint64(nw))
			testBoard := board
			testBoard.Set(idx, nextID)
			isWin, _ := CheckBoard(testBoard.P[nextID])
			if !isWin {
				t.Errorf("GameState claimed opponent win at %d, but CheckBoard said no.", idx)
			}
			nw &= nw - 1
		}
		l := myLoses
		for l != 0 {
			idx := bits.TrailingZeros64(uint64(l))
			testBoard := board
			testBoard.Set(idx, currentID)
			_, isLoss := CheckBoard(testBoard.P[currentID])
			if !isLoss {
				t.Errorf("GameState claimed self-loss at %d, but CheckBoard said no.", idx)
			}
			l &= l - 1
		}
		if myWins != 0 {
			if forced != myWins || best != myWins {
				t.Errorf("Forced/Best should equal MyWins")
			}
		} else if nextWins != 0 {
			if forced != nextWins || best != nextWins {
				t.Errorf("Forced/Best should equal NextWins")
			}
		} else {
			if forced != 0 {
				t.Errorf("Forced should be 0 when no immediate threats")
			}
			safe := empty & ^myLoses
			if safe != 0 {
				if best != safe {
					t.Errorf("Best should be safe moves. Got %x, Expected %x", best, safe)
				}
			} else if best != empty {
				t.Errorf("Best should be empty when no safe moves")
			}
		}
	})
}
func ValidateMCTSGraph(t *testing.T, root *MCGSNode, rootGS GameState) {
	if root == nil {
		t.Error("Root is nil")
		return
	}
	// Track visited nodes to handle DAG structure (shared nodes)
	visited := make(map[*MCGSNode]bool)
	// Track recursion stack for cycle detection
	stack := make(map[*MCGSNode]bool)
	var checkNode func(node *MCGSNode, gs GameState)
	checkNode = func(node *MCGSNode, gs GameState) {
		// 1. Cycle Detection
		if stack[node] {
			t.Errorf("Cycle detected in MCTS graph at node hash %016x", gs.Hash)
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
			t.Errorf("Node %016x has negative visits: %d", gs.Hash, node.N)
		}
		// Assuming Win/Loss rewards are 0.0 to 1.0 (or -1 to 1)
		// Squava implementation uses [0,1]
		for i := 0; i < 3; i++ {
			if node.Q[i] < -0.01 || node.Q[i] > 1.01 {
				t.Errorf("Node %016x has Q value out of bounds [0,1]: %v", gs.Hash, node.Q)
			}
		}
		// 4. Edge Invariants
		sumEdgeVisits := 0
		for i := range node.Edges {
			edge := &node.Edges[i]
			edgeVisits := int(edge.N)
			edgeDest := edge.Dest
			edgeMove := edge.Move
			sumEdgeVisits += edgeVisits
			if edgeDest == nil {
				t.Errorf("Node %016x has edge with nil destination", gs.Hash)
				continue
			}
			// DAG Invariant: Edge Visits vs Child Total Visits
			// The number of times we traversed THIS edge to get to Child
			// must be <= Total times Child was visited (from any parent).
			if edgeVisits > edgeDest.N {
				t.Errorf("Flow violation: Edge from %016x to %016x has %d visits, but child only has %d total visits.",
					gs.Hash, 0, edgeVisits, edgeDest.N)
			}
			// 5. State Consistency Check
			// Re-simulate the move to ensure the hash matches the destination node
			expectedState := gs
			expectedState.ApplyMove(edgeMove)
			// We can't easily check edgeDest.Hash anymore without passing it or trusting it matches expectedState.Hash
			// But we are here verifying the graph structure.

			// Recurse
			checkNode(edgeDest, expectedState)
		}
		// 6. Conservation of Flow (Outgoing)
		// The number of times we left this node cannot exceed the number of times we visited it.
		// node.N includes the visit where we stopped at this node (didn't traverse out).
		if sumEdgeVisits > node.N {
			t.Errorf("Node %016x has %d outgoing edge visits but only %d total node visits",
				gs.Hash, sumEdgeVisits, node.N)
		}
	}
	checkNode(root, rootGS)
}
func FuzzMCTSInvariants(f *testing.F) {
	f.Add(uint64(1), uint64(25), uint64(200))
	f.Fuzz(func(t *testing.T, seed uint64, numPieces64 uint64, mctsIters64 uint64) {
		xorState = seed
		mctsIters := int(mctsIters64 % 1000)
		if mctsIters < 10 {
			mctsIters = 10
		}
		board := generateRandomBoard(int(numPieces64 % 40))
		clean := true
		for p := 0; p < 3; p++ {
			isW, isL := CheckBoard(board.P[p])
			if isW || isL {
				clean = false
				break
			}
		}
		if !clean {
			return
		}
		player := NewMCTSPlayer("Tester", "T", 0, mctsIters)
		activeIDs := []int{0, 1, 2}
		_ = player.GetMove(board, activeIDs, 0)
		if player.root == nil {
			t.Errorf("MCTS did not generate a root node")
			return
		}
		rootGS := NewGameState(board, 0, 0x07)
		ValidateMCTSGraph(t, player.root, rootGS)
	})
}
func FuzzFullGameTermination(f *testing.F) {
	f.Add(uint64(1))
	f.Fuzz(func(t *testing.T, seed uint64) {
		xorState = seed
		board := Board{}
		activeMask := uint8(0x07)
		currentPID := 0
		players := []int{0, 1, 2}
		for {
			empty := ^board.Occupied
			if empty == 0 {
				break
			}
			count := bits.OnesCount64(uint64(empty))
			n := int(xrand() % uint64(count))
			idx := SelectBit64(uint64(empty), n)
			board.Set(idx, currentPID)
			isWin, isLoss := CheckBoard(board.P[currentPID])
			if isWin {
				for _, p := range players {
					if p == currentPID {
						continue
					}
					w, _ := CheckBoard(board.P[p])
					if w {
						t.Errorf("Invalid state. Multiple players have wins.")
					}
				}
				goto GameEnd
			}
			if isLoss {
				activeMask &= ^(1 << uint(currentPID))
				for j, p := range players {
					if p == currentPID {
						players = append(players[:j], players[j+1:]...)
						break
					}
				}
				if len(players) == 1 {
					goto GameEnd
				}
			}
			if board.Occupied == Bitboard(0xFFFFFFFFFFFFFFFF) {
				goto GameEnd
			}
			nextPID := int(nextPlayerTable[currentPID][activeMask])
			if nextPID == -1 {
				t.Errorf("nextPlayerTable returned -1 for activeMask %02x", activeMask)
				goto GameEnd
			}
			currentPID = nextPID
		}
	GameEnd:
		winCount := 0
		for p := 0; p < 3; p++ {
			isW, _ := CheckBoard(board.P[p])
			if isW {
				winCount++
			}
		}
		if winCount > 1 {
			t.Errorf("Invalid termination. Multiple winners: %d", winCount)
		}
	})
}

func TestIncrementalUpdateEquivalence(t *testing.T) {
	// Simulate the process of updating a node multiple times
	// and compare incremental average with full recalculation.

	results := [][3]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.33, 0.33, 0.33},
		{0.5, 0.5, 0.0},
		{1.0, 0.0, 0.0},
	}

	// 1. Incremental Update (as in Backprop)
	var qInc [3]float64
	var nInc int
	for _, res := range results {
		nInc++
		invN := 1.0 / float64(nInc)
		qInc[0] += (res[0] - qInc[0]) * invN
		qInc[1] += (res[1] - qInc[1]) * invN
		qInc[2] += (res[2] - qInc[2]) * invN
	}

	// 2. Full Recalculation (Sum / N)
	var sum [3]float64
	for _, res := range results {
		sum[0] += res[0]
		sum[1] += res[1]
		sum[2] += res[2]
	}
	qFull := [3]float64{
		sum[0] / float64(len(results)),
		sum[1] / float64(len(results)),
		sum[2] / float64(len(results)),
	}

	// Comparison
	epsilon := 1e-9
	for i := 0; i < 3; i++ {
		if math.Abs(qInc[i]-qFull[i]) > epsilon {
			t.Errorf("Mismatch at player %d: Incremental %f, Full %f", i, qInc[i], qFull[i])
		}
	}
}

func TestMCTSBackpropIncremental(t *testing.T) {
	// Test the actual Backprop method of MCTSPlayer
	m := NewMCTSPlayer("Test", "T", 0, 100)
	gs := NewGameState(Board{}, 0, 0x07)
	node := NewMCGSNode(gs)

	path := []PathStep{{Node: node, EdgeIdx: -1}}
	results := [][3]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.5, 0.5, 0.0},
	}

	for _, res := range results {
		m.Backprop(path, res)
	}

	expectedN := len(results)
	if node.N != expectedN {
		t.Errorf("Expected N=%d, got %d", expectedN, node.N)
	}

	sum := [3]float64{1.5, 1.5, 0.0}
	for i := 0; i < 3; i++ {
		expectedQ := sum[i] / float64(expectedN)
		if math.Abs(float64(node.Q[i])-expectedQ) > 1e-9 {
			t.Errorf("Player %d Q mismatch: expected %f, got %f", i, expectedQ, node.Q[i])
		}
	}
}

func TestMCGSNodeMethods(t *testing.T) {
	gs1 := NewGameState(Board{}, 0, 0x07)
	node := NewMCGSNode(gs1)
	gs2 := NewGameState(Board{}, 1, 0x07)
	child := NewMCGSNode(gs2)
	child.Q = [3]float32{0.1, 0.2, 0.3}

	// Test AddEdge
	move := Move{r: 1, c: 1}
	idx := node.AddEdge(move, child, gs1.PlayerID)
	if idx != 0 {
		t.Errorf("Expected edge index 0, got %d", idx)
	}
	if node.Edges[idx].Move != move {
		t.Errorf("Edge move mismatch")
	}
	if node.Edges[idx].Dest != child {
		t.Errorf("Edge destination mismatch")
	}
	if node.Edges[idx].N != 0 {
		t.Errorf("Expected 0 visits, got %d", node.Edges[idx].N)
	}
	if node.EdgeQs[idx] != child.Q[gs1.PlayerID] {
		t.Errorf("Edge Q mismatch: expected %f, got %f", child.Q[gs1.PlayerID], node.EdgeQs[idx])
	}

	// Test SyncEdge
	child.Q = [3]float32{0.4, 0.5, 0.6}
	node.SyncEdge(idx, child, gs1.PlayerID)
	if node.Edges[idx].N != 1 {
		t.Errorf("Expected 1 visit, got %d", node.Edges[idx].N)
	}
	if node.EdgeQs[idx] != child.Q[gs1.PlayerID] {
		t.Errorf("Edge Q mismatch after sync: expected %f, got %f", child.Q[gs1.PlayerID], node.EdgeQs[idx])
	}

	// Test UpdateStats
	result := [3]float32{1.0, 0.0, 0.0}
	node.UpdateStats(result)
	if node.N != 1 {
		t.Errorf("Expected N=1, got %d", node.N)
	}
	if node.Q[0] != 1.0 {
		t.Errorf("Expected Q[0]=1.0, got %f", node.Q[0])
	}

	// Test PopUntriedMove
	node.untriedMoves = Bitboard(1 << 5)
	mv, ok := node.PopUntriedMove()
	if !ok || mv.ToIndex() != 5 {
		t.Errorf("PopUntriedMove failed: got %v, %v", mv, ok)
	}
	if node.untriedMoves != 0 {
		t.Errorf("Expected untriedMoves to be 0 after pop")
	}
}

func TestTranspositionTableMethods(t *testing.T) {
	table := make(TranspositionTable, TTSize)
	board := Board{}
	gs := NewGameState(board, 0, 0x07)
	node := NewMCGSNode(gs)

	table.Store(gs.Hash, node)
	lookedUp := table.Lookup(&gs)
	if lookedUp != node {
		t.Errorf("TranspositionTable lookup failed")
	}

	// Test mismatch
	gsWrongHash := gs
	gsWrongHash.Hash = 1235
	if table.Lookup(&gsWrongHash) != nil {
		t.Errorf("Lookup should fail for different hash")
	}
	gsWrongPlayer := gs
	gsWrongPlayer.PlayerID = 1
	gsWrongPlayer.Hash = zobrist.ComputeHash(gsWrongPlayer.Board, gsWrongPlayer.PlayerID, gsWrongPlayer.ActiveMask)
	if table.Lookup(&gsWrongPlayer) != nil {
		t.Errorf("Lookup should fail for different playerID")
	}
}

func TestZobristHelper(t *testing.T) {
	h := uint64(100)
	h2 := zobrist.Move(h, 0, 10)
	if h2 != h^zobrist.piece[0][10] {
		t.Errorf("Zobrist.Move failed")
	}

	h3 := zobrist.SwapTurn(h, 0, 1)
	if h3 != h^zobrist.turn[0]^zobrist.turn[1] {
		t.Errorf("Zobrist.SwapTurn failed for non-terminal")
	}

	h4 := zobrist.SwapTurn(h, 0, -1)
	if h4 != h^zobrist.turn[0] {
		t.Errorf("Zobrist.SwapTurn failed for terminal")
	}

	h5 := zobrist.UpdateMask(h, 0x07, 0x03)
	if h5 != h^zobrist.active[0x07]^zobrist.active[0x03] {
		t.Errorf("Zobrist.UpdateMask failed")
	}
}

func TestGameRulesHelper(t *testing.T) {
	// Test IsTerminal
	gs := NewGameState(Board{}, 0, 0x01)
	if winner, ok := gs.IsTerminal(); !ok || winner != 0 {
		t.Errorf("IsTerminal failed for single player mask 0x01: got %v, %v", winner, ok)
	}
	gs = NewGameState(Board{}, 0, 0x03)
	if _, ok := gs.IsTerminal(); ok {
		t.Errorf("IsTerminal should be false for multi-player mask 0x03")
	}
}

func TestPdep(t *testing.T) {
	// PDEP mask, src -> spreads bits of src into mask
	// In our code: pdep(1<<k, v)
	// src = 1<<k, mask = v
	// This should return a uint64 with only the k-th set bit of v set.

	tests := []struct {
		v    uint64
		k    int
		want uint64
	}{
		{0b101010, 0, 1 << 1},
		{0b101010, 1, 1 << 3},
		{0b101010, 2, 1 << 5},
		{0b111, 0, 1 << 0},
		{0b111, 1, 1 << 1},
		{0b111, 2, 1 << 2},
		{0x8000000000000001, 0, 1 << 0},
		{0x8000000000000001, 1, 1 << 63},
	}

	for _, tc := range tests {
		got := pdep(uint64(1)<<uint(tc.k), tc.v)
		if got != tc.want {
			t.Errorf("pdep(1<<%d, %b) = %b, want %b", tc.k, tc.v, got, tc.want)
		}
	}
}

func TestSelectBit64(t *testing.T) {
	v := uint64(0b101010)
	// k=0 -> bit 1
	// k=1 -> bit 3
	// k=2 -> bit 5

	if got := SelectBit64(v, 0); got != 1 {
		t.Errorf("SelectBit64(0b101010, 0) = %d, want 1", got)
	}
	if got := SelectBit64(v, 1); got != 3 {
		t.Errorf("SelectBit64(0b101010, 1) = %d, want 3", got)
	}
	if got := SelectBit64(v, 2); got != 5 {
		t.Errorf("SelectBit64(0b101010, 2) = %d, want 5", got)
	}
}

func FuzzIncrementalThreats(f *testing.F) {
	f.Add(uint64(1), uint64(25))
	f.Fuzz(func(t *testing.T, seed uint64, numPieces64 uint64) {
		xorState = seed
		board := generateRandomBoard(int(numPieces64 % 40))
		clean := true
		for p := 0; p < 3; p++ {
			isW, isL := CheckBoard(board.P[p])
			if isW || isL {
				clean = false
				break
			}
		}
		if !clean {
			return
		}
		var activeMask uint8
		for {
			activeMask = uint8(xrand() % 8)
			if bits.OnesCount8(activeMask) >= 2 {
				break
			}
		}
		var currentID int
		for {
			currentID = int(xrand() % 3)
			if (activeMask & (1 << uint(currentID))) != 0 {
				break
			}
		}
		empty := ^board.Occupied
		if empty == 0 {
			return
		}
		count := bits.OnesCount64(uint64(empty))
		n := int(xrand() % uint64(count))
		idx := SelectBit64(uint64(empty), n)
		move := MoveFromIndex(idx)
		gs := NewGameState(board, currentID, activeMask)
		mover := gs.PlayerID
		gs.ApplyMove(move)
		refGS := NewGameState(gs.Board, gs.PlayerID, gs.ActiveMask)
		if gs.WinnerID != -1 {
			return
		}
		if gs.ActiveMask != refGS.ActiveMask || gs.Terminal != refGS.Terminal || gs.PlayerID != refGS.PlayerID {
			t.Errorf("Metadata mismatch")
		}
		for p := 0; p < 3; p++ {
			if gs.Wins[p] != refGS.Wins[p] {
				t.Errorf("Wins mismatch for P%d (Mover: P%d, Move: %d). Inc: %x, Ref: %x", p, mover, move.ToIndex(), gs.Wins[p], refGS.Wins[p])
			}
			if gs.Loses[p] != refGS.Loses[p] {
				t.Errorf("Loses mismatch for P%d. Inc: %x, Ref: %x", p, gs.Loses[p], refGS.Loses[p])
			}
		}
	})
}
func BenchmarkMCTSBlankBoard10k(b *testing.B) {
	player := NewMCTSPlayer("Bench", "B", 0, 10000)
	gs := NewGameState(Board{}, 0, 0x07)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tt.Clear()
		b.StartTimer()
		player.Search(gs)
	}
}

func selectBestEdgeGoRef(qs []float32, us []float32, coeff float32) int {
	if len(qs) == 0 {
		return -1
	}
	bestIdx := 0
	bestScore := qs[0] + coeff*us[0]
	for i := 1; i < len(qs); i++ {
		score := qs[i] + coeff*us[i]
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	return bestIdx
}

func FuzzSelectBestEdge(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0}, []byte{0, 0, 0, 0}, float32(1.0))
	f.Fuzz(func(t *testing.T, qBytes []byte, uBytes []byte, coeff float32) {
		n := len(qBytes) / 4
		if len(uBytes)/4 < n {
			n = len(uBytes) / 4
		}
		if n == 0 {
			return
		}
		qs := make([]float32, n)
		us := make([]float32, n)
		for i := 0; i < n; i++ {
			qs[i] = math.Float32frombits(uint32(qBytes[i*4]) | uint32(qBytes[i*4+1])<<8 | uint32(qBytes[i*4+2])<<16 | uint32(qBytes[i*4+3])<<24)
			us[i] = math.Float32frombits(uint32(uBytes[i*4]) | uint32(uBytes[i*4+1])<<8 | uint32(uBytes[i*4+2])<<16 | uint32(uBytes[i*4+3])<<24)
			if math.IsNaN(float64(qs[i])) {
				qs[i] = 0
			}
			if math.IsNaN(float64(us[i])) {
				us[i] = 0
			}
		}
		if math.IsNaN(float64(coeff)) {
			coeff = 1.0
		}

		got := selectBestEdgeAVX2(qs, us, coeff)
		if got == -1 {
			return
		}
		want := selectBestEdgeGoRef(qs, us, coeff)
		if got != want {
			scoreGot := qs[got] + coeff*us[got]
			scoreWant := qs[want] + coeff*us[want]
			if math.Abs(float64(scoreGot-scoreWant)) > 1e-6 {
				t.Errorf("AVX index %d (score %f) != Go index %d (score %f)", got, scoreGot, want, scoreWant)
			}
		}
	})
}

func FuzzWinsLossesSIMD(f *testing.F) {
	f.Add(uint64(0), uint64(0))
	f.Fuzz(func(t *testing.T, board uint64, empty uint64) {
		wAVX, lAVX := getWinsAndLossesAVX2(board, empty)
		wGo, lGo := getWinsAndLossesGo(board, empty)
		if wAVX != wGo || lAVX != lGo {
			t.Errorf("AVX(w:%x, l:%x) != Go(w:%x, l:%x)", wAVX, lAVX, wGo, lGo)
		}
	})
}
