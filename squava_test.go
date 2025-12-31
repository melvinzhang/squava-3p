package main

import (
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
						if (nr == r && nc == c) || (b & (1 << uint(nr*8+nc))) != 0 {
							count++
						}
					}
				}
				if count == 4 {
					isWin = true
					break
				}
			}
			if isWin { break }
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
						if (nr == r && nc == c) || (b & (1 << uint(nr*8+nc))) != 0 {
							count++
						}
					}
				}
				if count == 3 {
					isLoss = true
					break
				}
			}
			if isLoss { break }
		}
		if isLoss {
			l |= (1 << uint(i))
		}
	}

	return Bitboard(w), Bitboard(l)
}

func TestGetWinsAndLossesRandomized(t *testing.T) {
	for seed := int64(0); seed < 100000; seed++ {
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
