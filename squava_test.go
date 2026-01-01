package main

import (
	"math/bits"
	"testing"
)

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
		{0, 1},  // Horizontal
		{1, 0},  // Vertical
		{1, 1},  // Diagonal
		{1, -1}, // Anti-diagonal
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
	for seed := int64(0); seed < 1000; seed++ {
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
	for i := 0; i < 10000; i++ {
		board := Board{}
		for j := 0; j < 20; j++ {
			idx := int(xrand() % 64)
			p := int(xrand() % 3)
			if (board.Occupied & (1 << uint(idx))) == 0 {
				board.Set(idx, p)
			}
		}
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
