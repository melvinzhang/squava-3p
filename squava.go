package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"runtime/pprof"
	"strconv"
	"strings"
	"time"
)

// --- Faster random number generation (xorshift64*) ---
var xorState uint64 = 1 // seed should be non-zero

func xrand() uint64 {
	xorState ^= xorState >> 12
	xorState ^= xorState << 25
	xorState ^= xorState >> 27
	return xorState * 0x2545F4914F6CDD1D
}

func randIntn(n int) int {
	if n <= 0 {
		return 0
	}
	// Lemire's fast alternative to modulo for bounded random numbers
	return int((uint64(uint32(xrand())) * uint64(n)) >> 32)
}

var (
	zobristP      [3][64]uint64
	zobristTurn   [3]uint64
	zobristActive [256]uint64
)

func initZobrist() {
	r := rand.New(rand.NewSource(42))
	for p := 0; p < 3; p++ {
		for i := 0; i < 64; i++ {
			zobristP[p][i] = r.Uint64()
		}
		zobristTurn[p] = r.Uint64()
	}
	for i := 0; i < 256; i++ {
		zobristActive[i] = r.Uint64()
	}
}

const (
	BoardSize = 8
)

// Bitboard constants
const (
	FileA uint64 = 0x0101010101010101
	FileH uint64 = 0x8080808080808080
	Full  uint64 = 0xFFFFFFFFFFFFFFFF

	MaskNotA   uint64 = 0xFEFEFEFEFEFEFEFE
	MaskNotH   uint64 = 0x7F7F7F7F7F7F7F7F
	MaskNotAB  uint64 = 0xFCFCFCFCFCFCFCFC
	MaskNotGH  uint64 = 0x3F3F3F3F3F3F3F3F
	MaskNotABC uint64 = 0xF8F8F8F8F8F8F8F8
	MaskNotFGH uint64 = 0x1F1F1F1F1F1F1F1F
)

type Board struct {
	P        [3]Bitboard
	Occupied Bitboard
}
type Bitboard uint64
type Player interface {
	GetMove(board Board, players []int, turnIdx int) Move
	Name() string
	Symbol() string
	ID() int // 0, 1, 2
}
type PlayerInfo struct {
	name   string
	symbol string
	id     int
}

func (p *PlayerInfo) Name() string   { return p.name }
func (p *PlayerInfo) Symbol() string { return p.symbol }
func (p *PlayerInfo) ID() int        { return p.id }

type Move struct {
	r, c int8
}

func (m Move) ToIndex() int {
	return int(m.r)*8 + int(m.c)
}
func MoveFromIndex(idx int) Move {
	return Move{r: int8(idx / 8), c: int8(idx % 8)}
}

// --- Bitboard Logic ---
func (b *Board) Set(idx int, pID int) {
	mask := Bitboard(uint64(1) << idx)
	b.P[pID] |= mask
	b.Occupied |= mask
}

func (b *Board) Move(pID int, idx int) Bitboard {
	mask := Bitboard(uint64(1) << uint(idx))
	b.P[pID] |= mask
	b.Occupied |= mask
	return mask
}
func (b *Board) GetPlayerBoard(pID int) Bitboard {
	return b.P[pID]
}

func CheckBoard(bb Bitboard) (isWin, isLoss bool) {
	wins, loses := GetWinsAndLosses(bb, bb)
	isWin = wins != 0
	isLoss = !isWin && loses != 0
	return
}

// GetWinsAndLosses calculates win and loss bitboards.
// This function is manually unrolled for performance.
func GetWinsAndLosses(bb Bitboard, empty Bitboard) (wins Bitboard, loses Bitboard) {
	b := uint64(bb)
	e := uint64(empty)
	var w, l uint64

	// Direction 0: Horizontal (s=1)
	{
		r1 := (b >> 1) & MaskNotH
		l1 := (b << 1) & MaskNotA
		r2 := (b >> 2) & MaskNotGH
		l2 := (b << 2) & MaskNotAB

		r1r2 := r1 & r2
		l1l2 := l1 & l2
		l |= e & (r1r2 | r1&l1 | l1l2)

		r3 := (b >> 3) & MaskNotFGH
		l3 := (b << 3) & MaskNotABC
		w |= e & (r1r2&(r3|l1) | l1l2&(r1|l3))
	}

	// Direction 1: Vertical (s=8)
	{
		r1 := (b >> 8)
		l1 := (b << 8)
		r2 := (b >> 16)
		l2 := (b << 16)

		r1r2 := r1 & r2
		l1l2 := l1 & l2
		l |= e & (r1r2 | r1&l1 | l1l2)

		r3 := (b >> 24)
		l3 := (b << 24)
		w |= e & (r1r2&(r3|l1) | l1l2&(r1|l3))
	}

	// Direction 2: Diagonal (s=9)
	{
		r1 := (b >> 9) & MaskNotH
		l1 := (b << 9) & MaskNotA
		r2 := (b >> 18) & MaskNotGH
		l2 := (b << 18) & MaskNotAB

		r1r2 := r1 & r2
		l1l2 := l1 & l2
		l |= e & (r1r2 | r1&l1 | l1l2)

		r3 := (b >> 27) & MaskNotFGH
		l3 := (b << 27) & MaskNotABC
		w |= e & (r1r2&(r3|l1) | l1l2&(r1|l3))
	}

	// Direction 3: Anti-diagonal (s=7)
	{
		r1 := (b >> 7) & MaskNotA
		l1 := (b << 7) & MaskNotH
		r2 := (b >> 14) & MaskNotAB
		l2 := (b << 14) & MaskNotGH

		r1r2 := r1 & r2
		l1l2 := l1 & l2
		l |= e & (r1r2 | r1&l1 | l1l2)

		r3 := (b >> 21) & MaskNotABC
		l3 := (b << 21) & MaskNotFGH
		w |= e & (r1r2&(r3|l1) | l1l2&(r1|l3))
	}

	return Bitboard(w), Bitboard(l & ^w)
}

// ThreatAnalysis holds pre-calculated win/loss bitboards for a given turn.
type ThreatAnalysis struct {
	MyWins   Bitboard
	MyLoses  Bitboard
	NextWins Bitboard
}

// AnalyzeThreats calculates the immediate win/loss threats for the current and next player.
func AnalyzeThreats(board Board, currentPID, nextPID int) ThreatAnalysis {
	empty := ^board.Occupied
	myWins, myLoses := GetWinsAndLosses(board.GetPlayerBoard(currentPID), empty)
	nextWins, _ := GetWinsAndLosses(board.GetPlayerBoard(nextPID), empty)
	return ThreatAnalysis{
		MyWins:   myWins,
		MyLoses:  myLoses,
		NextWins: nextWins,
	}
}

func GetForcedMoves(board Board, currentPID, nextPID int) Bitboard {
	threats := AnalyzeThreats(board, currentPID, nextPID)
	if threats.MyWins != 0 {
		return threats.MyWins
	}
	return threats.NextWins
}

func GetBestMoves(board Board, threats ThreatAnalysis) Bitboard {
	if threats.MyWins != 0 {
		return threats.MyWins
	}
	if threats.NextWins != 0 {
		return threats.NextWins
	}

	empty := ^board.Occupied
	safeMoves := empty & ^threats.MyLoses
	if safeMoves != 0 {
		return safeMoves
	}
	return empty
}

// --- Human Player ---
type HumanPlayer struct {
	info PlayerInfo
}

func NewHumanPlayer(name, symbol string, id int) *HumanPlayer {
	return &HumanPlayer{info: PlayerInfo{name: name, symbol: symbol, id: id}}
}
func (h *HumanPlayer) Name() string   { return h.info.name }
func (h *HumanPlayer) Symbol() string { return h.info.symbol }
func (h *HumanPlayer) ID() int        { return h.info.id }
func (h *HumanPlayer) GetMove(board Board, players []int, turnIdx int) Move {
	nextPlayerIdx := (turnIdx + 1) % len(players)
	nextPID := players[nextPlayerIdx]
	forcedMoves := GetForcedMoves(board, h.info.id, nextPID)
	reader := bufio.NewReader(os.Stdin)
	for {
		prompt := fmt.Sprintf("%s (%s), enter your move (e.g., A1): ", h.info.name, h.info.symbol)
		if forcedMoves != 0 {
			forcedStr := []string{}
			temp := forcedMoves
			for temp != 0 {
				idx := bits.TrailingZeros64(uint64(temp))
				m := MoveFromIndex(idx)
				forcedStr = append(forcedStr, fmt.Sprintf("%c%d", int(m.c)+65, int(m.r)+1))
				temp &= Bitboard(^(uint64(1) << idx))
			}
			fmt.Printf("FORCED MOVE! You must block the next player. Valid moves: %s\n", strings.Join(forcedStr, ", "))
		}
		fmt.Print(prompt)
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(strings.ToUpper(input))
		r, c, err := parseInput(input)
		if err != nil {
			fmt.Println("Invalid format. Use algebraic (A1).")
			continue
		}
		if !isValidCoord(r, c) {
			fmt.Println("Move out of bounds.")
			continue
		}
		idx := r*8 + c
		mask := uint64(1) << idx
		if (uint64(board.Occupied) & mask) != 0 {
			fmt.Println("Cell already occupied.")
			continue
		}
		move := Move{int8(r), int8(c)}
		if forcedMoves != 0 && (forcedMoves&(Bitboard(1)<<idx)) == 0 {
			fmt.Println("Invalid move. You must block the opponent or win immediately.")
			continue
		}
		return move
	}
}
func parseInput(inp string) (int, int, error) {
	if len(inp) < 2 {
		return 0, 0, fmt.Errorf("invalid length")
	}
	colChar := inp[0]
	rowStr := inp[1:]
	if colChar < 'A' || colChar > 'H' {
		return 0, 0, fmt.Errorf("invalid column")
	}
	col := int(colChar - 'A')
	row, err := strconv.Atoi(rowStr)
	if err != nil {
		return 0, 0, err
	}
	return row - 1, col, nil
}
func isValidCoord(r, c int) bool {
	return r >= 0 && r < BoardSize && c >= 0 && c < BoardSize
}

var (
	invSqrtTable    [100000]float32
	coeffTable      [100000]float32
	tt              TranspositionTable
	select8         [256][8]uint8
	nextPlayerTable [3][256]int8
)

type Zobrist struct{}

func (Zobrist) Move(h uint64, pID int, idx int) uint64 {
	return h ^ zobristP[pID][idx]
}

func (Zobrist) SwapTurn(h uint64, oldPID, newPID int) uint64 {
	if newPID == -1 {
		return h ^ zobristTurn[oldPID]
	}
	return h ^ zobristTurn[oldPID] ^ zobristTurn[newPID]
}

func (Zobrist) UpdateMask(h uint64, oldMask, newMask uint8) uint64 {
	return h ^ zobristActive[oldMask] ^ zobristActive[newMask]
}

var zobrist Zobrist

type GameRules struct{}

func (GameRules) IsTerminal(mask uint8) (int, bool) {
	if bits.OnesCount8(mask) == 1 {
		return bits.TrailingZeros8(mask), true
	}
	return -1, false
}

func (GameRules) ResolveLoss(mask uint8, currentID int) (newMask uint8, winnerID int) {
	newMask = mask & ^(1 << uint(currentID))
	winnerID = -1
	if bits.OnesCount8(newMask) == 1 {
		winnerID = bits.TrailingZeros8(newMask)
	}
	return newMask, winnerID
}

var rules GameRules

func init() {
	for i := 0; i < 256; i++ {
		k := 0
		for j := 0; j < 8; j++ {
			if (i & (1 << uint(j))) != 0 {
				select8[i][k] = uint8(j)
				k++
			}
		}
	}

	initZobrist()
	for p := 0; p < 3; p++ {
		for m := 0; m < 256; m++ {
			nextPlayerTable[p][m] = -1
			for i := 1; i <= 2; i++ {
				next := (p + i) % 3
				if (m & (1 << uint(next))) != 0 {
					nextPlayerTable[p][m] = int8(next)
					break
				}
			}
		}
	}
	for i := 1; i < len(invSqrtTable); i++ {
		invSqrtTable[i] = float32(1.0 / math.Sqrt(float64(i)))
	}
	for i := 1; i < len(coeffTable); i++ {
		coeffTable[i] = float32(2.0 * math.Sqrt(math.Log(float64(i))))
	}
	tt = make(TranspositionTable, TTSize)
}

func getNextPlayer(currentID int, activeMask uint8) int {
	return int(nextPlayerTable[currentID][activeMask])
}

// --- MCTS Player ---
const TTSize = 1 << 24 // ~16M entries
const TTMask = TTSize - 1

type GameState struct {
	Board      Board
	Hash       uint64
	PlayerID   int
	ActiveMask uint8
	WinnerID   int
}

func (gs GameState) NextPlayer() int {
	return int(nextPlayerTable[gs.PlayerID][gs.ActiveMask])
}

func (gs GameState) IsTerminal() (int, bool) {
	if gs.WinnerID != -1 {
		return gs.WinnerID, true
	}
	return rules.IsTerminal(gs.ActiveMask)
}

func (gs GameState) GetBestMoves() Bitboard {
	nextP := gs.NextPlayer()
	threats := AnalyzeThreats(gs.Board, gs.PlayerID, nextP)
	return GetBestMoves(gs.Board, threats)
}

func (gs *GameState) applyPiece(idx int) {
	gs.Board.Move(gs.PlayerID, idx)
	gs.Hash = zobrist.Move(gs.Hash, gs.PlayerID, idx)
}

func (gs *GameState) updateTurn(nextID int) {
	gs.Hash = zobrist.SwapTurn(gs.Hash, gs.PlayerID, nextID)
	gs.PlayerID = nextID
}

func (gs *GameState) updateActiveMask(newMask uint8) {
	gs.Hash = zobrist.UpdateMask(gs.Hash, gs.ActiveMask, newMask)
	gs.ActiveMask = newMask
}

func (gs *GameState) setWinner(winnerID int) {
	gs.WinnerID = winnerID
	gs.Hash = zobrist.SwapTurn(gs.Hash, gs.PlayerID, -1)
	gs.PlayerID = -1
}

func (gs GameState) ApplyMove(move Move) GameState {
	newGS := gs
	idx := move.ToIndex()
	newGS.applyPiece(idx)

	isWin, isLoss := CheckBoard(newGS.Board.GetPlayerBoard(gs.PlayerID))
	if isWin {
		newGS.setWinner(gs.PlayerID)
		return newGS
	}
	if isLoss {
		newMask, winnerID := rules.ResolveLoss(gs.ActiveMask, gs.PlayerID)
		newGS.updateActiveMask(newMask)
		if winnerID != -1 {
			newGS.setWinner(winnerID)
		} else {
			newGS.updateTurn(getNextPlayer(gs.PlayerID, newMask))
		}
		return newGS
	}
	newGS.updateTurn(gs.NextPlayer())
	return newGS
}

type TTEntry struct {
	hash uint64
	node *MCGSNode
}

type TranspositionTable []TTEntry

func (tt TranspositionTable) Lookup(gs GameState) *MCGSNode {
	idx := gs.Hash & TTMask
	entry := tt[idx]
	if entry.hash == gs.Hash && entry.node != nil {
		if entry.node.Matches(gs) {
			return entry.node
		}
	}
	return nil
}

func (tt TranspositionTable) Store(hash uint64, node *MCGSNode) {
	idx := hash & TTMask
	tt[idx] = TTEntry{hash: hash, node: node}
}

type MCTSPlayer struct {
	info       PlayerInfo
	iterations int
	path       []PathStep
	root       *MCGSNode
	Verbose    bool
}

func NewMCTSPlayer(name, symbol string, id int, iterations int) *MCTSPlayer {
	return &MCTSPlayer{
		info:       PlayerInfo{name: name, symbol: symbol, id: id},
		iterations: iterations,
		path:       make([]PathStep, 0, 64),
	}
}
func (m *MCTSPlayer) Name() string   { return m.info.name }
func (m *MCTSPlayer) Symbol() string { return m.info.symbol }
func (m *MCTSPlayer) ID() int        { return m.info.id }

func ZobristHash(board Board, playerToMoveID int, activeMask uint8) uint64 {
	var h uint64
	if playerToMoveID >= 0 && playerToMoveID < 3 {
		h = zobristTurn[playerToMoveID]
	}
	h ^= zobristActive[activeMask]
	for p := 0; p < 3; p++ {
		pBoard := uint64(board.P[p])
		for pBoard != 0 {
			idx := bits.TrailingZeros64(pBoard)
			h ^= zobristP[p][idx]
			pBoard &= pBoard - 1
		}
	}
	return h
}

func (m *MCTSPlayer) Search(gs GameState) (int, int) {
	root := tt.Lookup(gs)
	if root == nil {
		root = NewMCGSNode(gs)
		tt.Store(gs.Hash, root)
	}
	m.root = root

	initialN := root.N
	totalSteps := 0
	for root.N < m.iterations {
		path := m.Select(root)
		leaf := path[len(path)-1].Node

		var result [3]float32
		if winnerID, ok := leaf.IsTerminal(); ok {
			result = ScoreWin(winnerID)
		} else {
			var s int
			result, s, _ = RunSimulation(leaf.GameState)
			totalSteps += s
		}
		m.Backprop(path, result)
	}
	return totalSteps, root.N - initialN
}

type MoveStat struct {
	mv      Move
	visits  int
	winrate float32
}

func (m *MCTSPlayer) PrintStats(myID int, totalSteps, rollouts int, elapsed time.Duration) {
	if !m.Verbose {
		return
	}
	root := m.root
	if elapsed.Seconds() > 0 {
		sps := float64(totalSteps) / elapsed.Seconds()
		fmt.Printf("Rollouts: %d, Steps: %d, Time: %v, SPS: %.2f\n", rollouts, totalSteps, elapsed, sps)
	}
	fmt.Printf("Estimated Winrate: %.2f%%\n", root.Q[myID]*100)

	stats := []MoveStat{}
	bestVisits := -1
	for i := range root.EdgeDests {
		mv := root.EdgeMoves[i]
		visits := int(root.EdgeVisits[i])
		q := root.EdgeDests[i].Q[myID]
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

func (m *MCTSPlayer) GetMove(board Board, players []int, turnIdx int) Move {
	activeMask := uint8(0)
	for _, pID := range players {
		activeMask |= 1 << uint(pID)
	}
	rootHash := ZobristHash(board, players[turnIdx], activeMask)
	gs := GameState{Board: board, Hash: rootHash, PlayerID: players[turnIdx], ActiveMask: activeMask, WinnerID: -1}

	startTime := time.Now()
	totalSteps, rollouts := m.Search(gs)
	elapsed := time.Since(startTime)

	m.PrintStats(players[turnIdx], totalSteps, rollouts, elapsed)

	bestVisits := -1
	var bestMove Move
	for i := range m.root.EdgeDests {
		visits := int(m.root.EdgeVisits[i])
		if visits > bestVisits {
			bestVisits = visits
			bestMove = m.root.EdgeMoves[i]
		}
	}

	if bestVisits == -1 {
		// Fallback
		moves := gs.GetBestMoves()
		if moves != 0 {
			idx := bits.TrailingZeros64(uint64(moves))
			return MoveFromIndex(idx)
		}
	}
	return bestMove
}

type PathStep struct {
	Node    *MCGSNode
	EdgeIdx int // Index in the parent's Edges slice
}

var negInf = math.Inf(-1)

func (m *MCTSPlayer) Select(root *MCGSNode) []PathStep {
	m.path = m.path[:0]
	m.path = append(m.path, PathStep{Node: root, EdgeIdx: -1})

	curr := root
	for {
		if _, terminal := curr.IsTerminal(); terminal {
			break
		}
		// --- Expansion Phase ---
		// If the current node has moves that haven't been added to the graph yet,
		// pick one at random and expand the search.
		if move, ok := curr.PopUntriedMove(); ok {
			// Determine the new state and look it up in the Transposition Table.
			state := curr.ApplyMove(move)

			child := tt.Lookup(state)
			isNew := child == nil

			if isNew {
				child = NewMCGSNode(state)
				tt.Store(state.Hash, child)
			}

			// Add the edge to the graph and record it in the selection path.
			edgeIdx := curr.AddEdge(move, child)
			m.path = append(m.path, PathStep{Node: child, EdgeIdx: edgeIdx})
			// If we've added a genuinely new node to the TT, we stop and proceed to rollout.
			// Otherwise, we continue selection from this existing node.
			if isNew {
				return m.path
			}
			curr = child
			continue
		}

		// --- Selection Phase ---
		// If all moves from the current node have been explored, use the UCB1
		// formula to select the most promising child for further exploration.
		if len(curr.EdgeDests) == 0 {
			break // Terminal node reached.
		}

		bestIdx := -1
		bestScore := float32(negInf)
		coeff := curr.UCB1Coeff
		pID := curr.PlayerID

		visits := curr.EdgeVisits
		qs := curr.EdgeQs[pID]
		for i := range visits {
			vPlus1 := int(visits[i]) + 1
			var u float32
			if vPlus1 < len(invSqrtTable) {
				u = coeff * invSqrtTable[vPlus1]
			} else {
				u = coeff / float32(math.Sqrt(float64(vPlus1)))
			}
			score := qs[i] + u
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break
		}

		m.path = append(m.path, PathStep{Node: curr.EdgeDests[bestIdx], EdgeIdx: bestIdx})
		curr = curr.EdgeDests[bestIdx]

	}
	return m.path
}
func (m *MCTSPlayer) Backprop(path []PathStep, result [3]float32) {
	for i := len(path) - 1; i >= 0; i-- {
		step := path[i]
		node := step.Node
		node.UpdateStats(result)

		if i > 0 && step.EdgeIdx != -1 {
			parent := path[i-1].Node
			parent.SyncEdge(step.EdgeIdx, node)
		}
	}
}

type MCGSNode struct {
	GameState
	N            int
	Q            [3]float32
	EdgeMoves    []Move
	EdgeDests    []*MCGSNode
	EdgeVisits   []int32
	EdgeQs       [3][]float32
	untriedMoves Bitboard
	UCB1Coeff    float32
}

func (n *MCGSNode) AddEdge(move Move, dest *MCGSNode) int {
	idx := len(n.EdgeDests)
	n.EdgeMoves = append(n.EdgeMoves, move)
	n.EdgeDests = append(n.EdgeDests, dest)
	n.EdgeVisits = append(n.EdgeVisits, 0)
	n.EdgeQs[0] = append(n.EdgeQs[0], dest.Q[0])
	n.EdgeQs[1] = append(n.EdgeQs[1], dest.Q[1])
	n.EdgeQs[2] = append(n.EdgeQs[2], dest.Q[2])
	return idx
}

func (n *MCGSNode) UpdateStats(result [3]float32) {
	n.N++
	invN := 1.0 / float32(n.N)
	n.Q[0] += (result[0] - n.Q[0]) * invN
	n.Q[1] += (result[1] - n.Q[1]) * invN
	n.Q[2] += (result[2] - n.Q[2]) * invN

	// Update cached coeff
	nPlus1 := n.N + 1
	if nPlus1 < len(coeffTable) {
		n.UCB1Coeff = coeffTable[nPlus1]
	} else {
		n.UCB1Coeff = float32(2.0 * math.Sqrt(math.Log(float64(nPlus1))))
	}
}

func (n *MCGSNode) SyncEdge(idx int, child *MCGSNode) {
	n.EdgeVisits[idx]++
	n.EdgeQs[0][idx] = child.Q[0]
	n.EdgeQs[1][idx] = child.Q[1]
	n.EdgeQs[2][idx] = child.Q[2]
}

func (n *MCGSNode) PopUntriedMove() (Move, bool) {
	moveIdx := PickRandomBit(n.untriedMoves)
	if moveIdx == -1 {
		return Move{}, false
	}
	n.untriedMoves &= ^(Bitboard(1) << uint(moveIdx))
	return MoveFromIndex(moveIdx), true
}

type MCGSEdge struct {
	Move   Move
	Dest   *MCGSNode
	Visits int
}

func NewMCGSNode(gs GameState) *MCGSNode {
	var untried Bitboard
	if _, terminal := gs.IsTerminal(); !terminal {
		untried = gs.GetBestMoves()
	}
	n := &MCGSNode{
		GameState:    gs,
		untriedMoves: untried,
	}
	return n
}
func (n *MCGSNode) Matches(gs GameState) bool {
	return n.GameState == gs
}

func pdep(src, mask uint64) uint64

func SelectBit64(v uint64, k int) int {
	return bits.TrailingZeros64(pdep(uint64(1)<<uint(k), v))
}

func PickRandomBit(bb Bitboard) int {
	count := bits.OnesCount64(uint64(bb))
	if count == 0 {
		return -1
	}
	if count == 1 {
		return bits.TrailingZeros64(uint64(bb))
	}
	hi, _ := bits.Mul64(xrand(), uint64(count))
	return SelectBit64(uint64(bb), int(hi))
}

func ScoreWin(winnerID int) [3]float32 {
	var res [3]float32
	if winnerID >= 0 && winnerID < 3 {
		res[winnerID] = 1.0
	}
	return res
}

func ScoreDraw(mask uint8) [3]float32 {
	var res [3]float32
	count := bits.OnesCount8(mask)
	if count == 0 {
		return res
	}
	score := 1.0 / float32(count)
	for p := 0; p < 3; p++ {
		if (mask & (1 << uint(p))) != 0 {
			res[p] = score
		}
	}
	return res
}

// --- Simulation Logic ---
func RunSimulation(gs GameState) ([3]float32, int, Board) {
	steps := 0
	for {
		steps++
		if winnerID, ok := gs.IsTerminal(); ok {
			return ScoreWin(winnerID), steps, gs.Board
		}

		moves := gs.GetBestMoves()
		idx := PickRandomBit(moves)
		if idx == -1 {
			return ScoreDraw(gs.ActiveMask), steps, gs.Board
		}

		gs = gs.ApplyMove(MoveFromIndex(idx))
	}
}

// --- Game Engine ---
type SquavaGame struct {
	board   Board
	players []Player
	turnIdx int
}

func NewSquavaGame() *SquavaGame {
	return &SquavaGame{}
}
func (g *SquavaGame) AddPlayer(p Player) {
	g.players = append(g.players, p)
}

func (g *SquavaGame) ActiveIDs() []int {
	activeIDs := make([]int, len(g.players))
	for i, p := range g.players {
		activeIDs[i] = p.ID()
	}
	return activeIDs
}

func (g *SquavaGame) RemovePlayer(idx int) bool {
	player := g.players[idx]
	fmt.Printf("Oops! %s made 3 in a row and is eliminated!\n", player.Name())
	g.players = append(g.players[:idx], g.players[idx+1:]...)
	if g.turnIdx >= len(g.players) {
		g.turnIdx = 0
	}
	return len(g.players) == 1
}

func (g *SquavaGame) PrintBoard() {
	fmt.Print("   ")
	for i := 0; i < BoardSize; i++ {
		fmt.Printf("%c ", 'A'+i)
	}
	fmt.Println()
	for r := 0; r < BoardSize; r++ {
		fmt.Printf("%2d ", r+1)
		for c := 0; c < BoardSize; c++ {
			symbol := "."
			idx := r*8 + c
			mask := Bitboard(uint64(1) << idx)
			if (g.board.P[0] & mask) != 0 {
				symbol = "X"
			} else if (g.board.P[1] & mask) != 0 {
				symbol = "O"
			} else if (g.board.P[2] & mask) != 0 {
				symbol = "Z"
			}
			fmt.Printf("%s ", symbol)
		}
		fmt.Println()
	}
}

func (g *SquavaGame) Run() {
	fmt.Println("Starting 3-Player Squava!")
	fmt.Printf("Random Seed: %d\n", xorState)
	fmt.Println("Board Size: 8x8")
	fmt.Println("Rules: 4-in-a-row wins. 3-in-a-row loses.")
	moveCount := 1
	for {
		if len(g.players) == 0 {
			fmt.Println("All players eliminated? Draw.")
			break
		}
		if len(g.players) == 1 {
			g.PrintBoard()
			fmt.Printf("%s wins as the last player standing!\n", g.players[0].Name())
			break
		}
		currentPlayer := g.players[g.turnIdx]

		g.PrintBoard()
		fmt.Printf("Move %d: %s (%s)\n", moveCount, currentPlayer.Name(), currentPlayer.Symbol())

		if _, ok := currentPlayer.(*MCTSPlayer); ok {
			fmt.Printf("%s is thinking...\n", currentPlayer.Name())
		}

		move := currentPlayer.GetMove(g.board, g.ActiveIDs(), g.turnIdx)

		if _, ok := currentPlayer.(*MCTSPlayer); ok {
			fmt.Printf("%s chooses %c%d\n", currentPlayer.Name(), int(move.c)+65, int(move.r)+1)
		}
		g.board.Set(move.ToIndex(), currentPlayer.ID())
		moveCount++
		isWin, isLoss := CheckBoard(g.board.GetPlayerBoard(currentPlayer.ID()))
		if isWin {
			g.PrintBoard()
			fmt.Printf("!!! %s wins with 4 in a row! !!!\n", currentPlayer.Name())
			return
		}
		if isLoss {
			g.PrintBoard()
			if g.RemovePlayer(g.turnIdx) {
				g.PrintBoard()
				fmt.Printf("%s wins as the last player standing!\n", g.players[0].Name())
				return
			}

			if g.board.Occupied == Bitboard(Full) {
				g.PrintBoard()
				fmt.Println("Board full! Game is a Draw between remaining players.")
				return
			}
			continue
		}
		if g.board.Occupied == Bitboard(Full) {
			g.PrintBoard()
			fmt.Println("Board full! Game is a Draw.")
			return
		}
		g.turnIdx = (g.turnIdx + 1) % len(g.players)
	}
}
func main() {
	p1Type := flag.String("p1", "human", "Player 1 type (human/mcts)")
	p2Type := flag.String("p2", "human", "Player 2 type (human/mcts)")
	p3Type := flag.String("p3", "human", "Player 3 type (human/mcts)")
	iterations := flag.Int("iterations", 1000, "MCTS iterations")
	cpuProfile := flag.String("cpuprofile", "", "write cpu profile to file")
	seed := flag.Int64("seed", 0, "Random seed (0 for time-based)")
	flag.Parse()

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "could not create CPU profile: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "could not start CPU profile: %v\n", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}
	if *seed == 0 {
		xorState = uint64(time.Now().UnixNano())
	} else {
		xorState = uint64(*seed)
	}
	if xorState == 0 {
		xorState = 1
	}
	game := NewSquavaGame()
	createPlayer := func(t, name, symbol string, id int) Player {
		if t == "mcts" {
			p := NewMCTSPlayer(name, symbol, id, *iterations)
			p.Verbose = true
			return p
		}
		return NewHumanPlayer(name, symbol, id)
	}
	game.AddPlayer(createPlayer(*p1Type, "Player 1", "X", 0))
	game.AddPlayer(createPlayer(*p2Type, "Player 2", "O", 1))
	game.AddPlayer(createPlayer(*p3Type, "Player 3", "Z", 2))
	game.Run()
}
