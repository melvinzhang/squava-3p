//go:build js && wasm

package main

import (
	"fmt"
	"math/bits"
	"syscall/js"
)

var currentGS GameState

func newGame(this js.Value, args []js.Value) any {
	if len(args) > 0 {
		seedStr := args[0].String()
		var s uint64
		fmt.Sscanf(seedStr, "%d", &s)
		if s == 0 {
			s = 1
		}
		xorState = s
	}
	board := Board{}
	activeMask := uint8(0x07) // All 3 players active
	currentGS = NewGameState(board, 0, activeMask)
	return js.ValueOf(fmt.Sprintf("%d", currentGS.Hash))
}

func applyMove(this js.Value, args []js.Value) any {
	if len(args) < 1 {
		return js.ValueOf(false)
	}
	idx := args[0].Int()
	mask := Bitboard(1 << uint(idx))

	if (currentGS.Board.Occupied & mask) != 0 {
		return js.ValueOf(false)
	}

	activeIDs := currentGS.ActiveIDs()
	var turnIdx int
	for i, id := range activeIDs {
		if id == currentGS.PlayerID {
			turnIdx = i
			break
		}
	}
	forced := GetForcedMoves(currentGS.Board, activeIDs, turnIdx)
	if forced != 0 && (forced&(Bitboard(1)<<uint(idx))) == 0 {
		return js.ValueOf(false)
	}

	move := MoveFromIndex(idx)
	currentGS.ApplyMove(move)
	return js.ValueOf(fmt.Sprintf("%d", currentGS.Hash))
}

func getForcedMoves(this js.Value, args []js.Value) any {
	activeIDs := currentGS.ActiveIDs()
	var turnIdx int
	for i, id := range activeIDs {
		if id == currentGS.PlayerID {
			turnIdx = i
			break
		}
	}
	forced := GetForcedMoves(currentGS.Board, activeIDs, turnIdx)
	return js.ValueOf(fmt.Sprintf("%d", forced))
}

func getBestMove(this js.Value, args []js.Value) any {
	iterations := 10000
	if len(args) > 0 {
		iterations = args[0].Int()
	}

	activeIDs := currentGS.ActiveIDs()
	var turnIdx int
	for i, id := range activeIDs {
		if id == currentGS.PlayerID {
			turnIdx = i
			break
		}
	}

	// Fast path for forced moves
	forced := GetForcedMoves(currentGS.Board, activeIDs, turnIdx)
	if forced != 0 && bits.OnesCount64(uint64(forced)) == 1 {
		return js.ValueOf(bits.TrailingZeros64(uint64(forced)))
	}

	player := NewMCTSPlayer("AI", "AI", currentGS.PlayerID, iterations)
	player.Verbose = false
	move := player.GetMove(currentGS.Board, activeIDs, turnIdx)
	return js.ValueOf(move.ToIndex())
}

func getBoard(this js.Value, args []js.Value) any {
	p0 := fmt.Sprintf("%d", currentGS.Board.P[0])
	p1 := fmt.Sprintf("%d", currentGS.Board.P[1])
	p2 := fmt.Sprintf("%d", currentGS.Board.P[2])

	activeIDs := currentGS.ActiveIDs()
	var turnIdx int
	for i, id := range activeIDs {
		if id == currentGS.PlayerID {
			turnIdx = i
			break
		}
	}
	forced := GetForcedMoves(currentGS.Board, activeIDs, turnIdx)

	res := js.Global().Get("Object").New()
	res.Set("p0", p0)
	res.Set("p1", p1)
	res.Set("p2", p2)
	res.Set("playerID", currentGS.PlayerID)
	res.Set("activeMask", int(currentGS.ActiveMask))
	res.Set("forcedMoves", fmt.Sprintf("%d", forced))
	winnerID, terminal := currentGS.IsTerminal()
	res.Set("winnerID", winnerID)
	res.Set("terminal", terminal)

	return res
}


func main() {
	c := make(chan struct{}, 0)
	fmt.Println("Squava Engine Initialized")
	js.Global().Set("squavaNewGame", js.FuncOf(newGame))
	js.Global().Set("squavaApplyMove", js.FuncOf(applyMove))
	js.Global().Set("squavaGetBestMove", js.FuncOf(getBestMove))
	js.Global().Set("squavaGetBoard", js.FuncOf(getBoard))
	js.Global().Set("squavaGetForcedMoves", js.FuncOf(getForcedMoves))
	<-c
}
