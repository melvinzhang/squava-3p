package main

import (
	"math"
	"math/bits"
)

// --- Faster random number generation (xorshift64*) ---
var xorState uint64 = 1 // seed should be non-zero

func xrand() uint64 {
	xorState ^= xorState >> 12
	xorState ^= xorState << 25
	xorState ^= xorState >> 27
	return xorState * 0x2545F4914F6CDD1D
}

type ZobristTable struct {
	piece  [3][64]uint64
	turn   [3]uint64
	active [256]uint64
}

func NewZobristTable() *ZobristTable {
	z := &ZobristTable{}
	// Use a local xorshift for deterministic initialization
	s := uint64(42)
	next := func() uint64 {
		s ^= s >> 12
		s ^= s << 25
		s ^= s >> 27
		return s * 0x2545F4914F6CDD1D
	}

	for p := 0; p < 3; p++ {
		for i := 0; i < 64; i++ {
			z.piece[p][i] = next()
		}
		z.turn[p] = next()
	}
	for i := 0; i < 256; i++ {
		z.active[i] = next()
	}
	return z
}

var zobrist *ZobristTable

const (
	BoardSize = 8
)

// Bitboard constants
const (
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
func GetWinsAndLosses(bb Bitboard, empty Bitboard) (wins Bitboard, loses Bitboard) {
	w, l := getWinsAndLossesAVX2(uint64(bb), uint64(empty))
	return Bitboard(w), Bitboard(l & ^w)
}

func getWinsAndLossesGo(b, e uint64) (w, l uint64) {
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

	return
}

func GetForcedMoves(board Board, players []int, turnIdx int) Bitboard {
	activeMask := uint8(0)
	for _, pID := range players {
		activeMask |= 1 << uint(pID)
	}
	gs := NewGameState(board, players[turnIdx], activeMask)

	if gs.Wins[gs.PlayerID] != 0 {
		return gs.Wins[gs.PlayerID]
	}
	nextP := gs.NextPlayer()
	if nextP != -1 {
		return gs.Wins[nextP]
	}
	return 0
}

var (
	invSqrtTable    [100000]float32
	coeffTable      [100000]float32
	tt              TranspositionTable
	nextPlayerTable [3][256]int8
)

func (z *ZobristTable) Move(h uint64, pID int, idx int) uint64 {
	return h ^ z.piece[pID][idx]
}

func (z *ZobristTable) SwapTurn(h uint64, oldPID, newPID int) uint64 {
	if newPID == -1 {
		return h ^ z.turn[oldPID]
	}
	return h ^ z.turn[oldPID] ^ z.turn[newPID]
}

func (z *ZobristTable) UpdateMask(h uint64, oldMask, newMask uint8) uint64 {
	return h ^ z.active[oldMask] ^ z.active[newMask]
}

func (z *ZobristTable) ComputeHash(board Board, playerToMoveID int, activeMask uint8) uint64 {
	var h uint64
	if playerToMoveID >= 0 && playerToMoveID < 3 {
		h = z.turn[playerToMoveID]
	}
	h ^= z.active[activeMask]
	for p := 0; p < 3; p++ {
		pBoard := uint64(board.P[p])
		for pBoard != 0 {
			idx := bits.TrailingZeros64(pBoard)
			h ^= z.piece[p][idx]
			pBoard &= pBoard - 1
		}
	}
	return h
}

func init() {
	zobrist = NewZobristTable()
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
		coeffTable[i] = float32(math.Sqrt(2.0 * math.Log(float64(i))))
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
	Terminal   bool
	Wins       [3]Bitboard
	Loses      [3]Bitboard
}

func NewGameState(board Board, playerID int, activeMask uint8) GameState {
	gs := GameState{
		Board:      board,
		PlayerID:   playerID,
		ActiveMask: activeMask,
		WinnerID:   -1,
	}
	gs.Hash = zobrist.ComputeHash(board, playerID, activeMask)
	gs.InitThreats()
	return gs
}

func (gs *GameState) NextPlayer() int {
	return int(nextPlayerTable[gs.PlayerID][gs.ActiveMask])
}

func (gs *GameState) IsTerminal() (int, bool) {
	return gs.WinnerID, gs.Terminal
}

func (gs *GameState) ActiveIDs() []int {
	ids := make([]int, 0, 3)
	for i := 0; i < 3; i++ {
		if (gs.ActiveMask & (1 << uint(i))) != 0 {
			ids = append(ids, i)
		}
	}
	return ids
}

func (gs *GameState) GetBestMoves() Bitboard {
	if gs.Wins[gs.PlayerID] != 0 {
		return gs.Wins[gs.PlayerID]
	}
	nextP := gs.NextPlayer()
	if nextP != -1 && gs.Wins[nextP] != 0 {
		return gs.Wins[nextP]
	}
	empty := ^gs.Board.Occupied
	safe := empty & ^gs.Loses[gs.PlayerID]
	if safe != 0 {
		return safe
	}
	return empty
}

func (gs *GameState) InitThreats() {
	empty := ^gs.Board.Occupied
	activeCount := bits.OnesCount8(gs.ActiveMask)

	// Re-evaluate terminal state
	if gs.WinnerID != -1 {
		gs.Terminal = true
	} else if activeCount <= 1 {
		gs.Terminal = true
		if activeCount == 1 {
			gs.WinnerID = bits.TrailingZeros8(gs.ActiveMask)
		}
	} else if empty == 0 {
		gs.Terminal = true
	} else {
		gs.Terminal = false
	}

	for p := 0; p < 3; p++ {
		if (gs.ActiveMask & (1 << uint(p))) != 0 {
			gs.Wins[p], gs.Loses[p] = GetWinsAndLosses(gs.Board.P[p], empty)
		} else {
			gs.Wins[p] = 0
			gs.Loses[p] = 0
		}
	}
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
	gs.Terminal = true
}

func (gs *GameState) ApplyMove(move Move) {
	gs.ApplyMoveIdx(move.ToIndex())
}

func (gs *GameState) ApplyMoveIdx(idx int) {
	mask := Bitboard(1 << uint(idx))
	pID := gs.PlayerID

	// 1. Immediate win
	if (gs.Wins[pID] & mask) != 0 {
		gs.applyPiece(idx)
		gs.setWinner(pID)
		return
	}

	// 2. Normal move or elimination
	isLoss := (gs.Loses[pID] & mask) != 0
	gs.applyPiece(idx)

	empty := ^gs.Board.Occupied
	invMask := ^mask

	if isLoss {
		newMask := gs.ActiveMask & ^(1 << uint(pID))
		gs.updateActiveMask(newMask)
		if bits.OnesCount8(newMask) == 1 {
			gs.setWinner(bits.TrailingZeros8(newMask))
		} else {
			gs.updateTurn(getNextPlayer(pID, newMask))
		}
		gs.Wins[pID] = 0
		gs.Loses[pID] = 0
	} else {
		gs.updateTurn(gs.NextPlayer())
		if empty == 0 {
			gs.Terminal = true
		} else {
			gs.Wins[pID], gs.Loses[pID] = GetWinsAndLosses(gs.Board.P[pID], empty)
		}
	}

	// Update other players' threats - unrolled loop
	if pID != 0 {
		gs.Wins[0] &= invMask
		gs.Loses[0] &= invMask
	}
	if pID != 1 {
		gs.Wins[1] &= invMask
		gs.Loses[1] &= invMask
	}
	if pID != 2 {
		gs.Wins[2] &= invMask
		gs.Loses[2] &= invMask
	}
}

type TTEntry struct {
	hash uint64
	node *MCGSNode
}

type TranspositionTable []TTEntry

func (tt TranspositionTable) Lookup(gs *GameState) *MCGSNode {
	idx := gs.Hash & TTMask
	entry := tt[idx]
	if entry.hash == gs.Hash && entry.node != nil {
		return entry.node
	}
	return nil
}

func (tt TranspositionTable) Store(hash uint64, node *MCGSNode) {
	idx := hash & TTMask
	tt[idx] = TTEntry{hash: hash, node: node}
}

func (tt TranspositionTable) Clear() {
	for i := range tt {
		tt[i] = TTEntry{}
	}
}

type MCTSPlayer struct {
	info       PlayerInfo
	iterations int
	root       *MCGSNode
	Verbose    bool
}

func NewMCTSPlayer(name, symbol string, id int, iterations int) *MCTSPlayer {
	return &MCTSPlayer{
		info:       PlayerInfo{name: name, symbol: symbol, id: id},
		iterations: iterations,
	}
}
func (m *MCTSPlayer) Name() string   { return m.info.name }
func (m *MCTSPlayer) Symbol() string { return m.info.symbol }
func (m *MCTSPlayer) ID() int        { return m.info.id }

func (m *MCTSPlayer) Search(gs GameState) (int, int) {
	root := tt.Lookup(&gs)
	if root == nil {
		root = NewMCGSNode(gs)
		tt.Store(gs.Hash, root)
	}
	m.root = root

	initialN := root.N
	totalSteps := 0
	path := make([]PathStep, 0, 64)
	for root.N < m.iterations {
		tmpGS := gs
		path = path[:0]
		path = m.Select(root, &tmpGS, path)

		var result [3]float32
		winnerID, terminal := tmpGS.IsTerminal()
		if terminal {
			result = ScoreTerminal(tmpGS.ActiveMask, winnerID)
		} else {
			var s int
			result, s, _ = RunSimulation(&tmpGS)
			totalSteps += s
		}
		m.Backprop(path, result)
	}
	return totalSteps, root.N - initialN
}

func (m *MCTSPlayer) GetMove(board Board, players []int, turnIdx int) Move {
	activeMask := uint8(0)
	for _, pID := range players {
		activeMask |= 1 << uint(pID)
	}
	gs := NewGameState(board, players[turnIdx], activeMask)

	totalSteps, rollouts := m.Search(gs)

	m.PrintStats(players[turnIdx], totalSteps, rollouts)

	bestVisits := -1
	var bestMove Move
	for i := range m.root.Edges {
		edge := &m.root.Edges[i]
		visits := int(edge.N)
		if visits > bestVisits {
			bestVisits = visits
			bestMove = edge.Move
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
	Node     *MCGSNode
	EdgeIdx  int // Index in the parent's Edges slice
	PlayerID int // Player who acts at Node
}

var negInf = math.Inf(-1)

func (m *MCTSPlayer) Select(root *MCGSNode, gs *GameState, path []PathStep) []PathStep {
	path = append(path, PathStep{Node: root, EdgeIdx: -1, PlayerID: gs.PlayerID})
	curr := root

	for {
		if _, terminal := gs.IsTerminal(); terminal {
			return path
		}

		if curr.untriedMoves != 0 {
			move, _ := curr.PopUntriedMove()
			child, _, edgeIdx := m.expand(curr, gs, move, gs.PlayerID)
			path = append(path, PathStep{Node: child, EdgeIdx: edgeIdx, PlayerID: gs.PlayerID})
			return path
		} else {
			bestIdx := curr.selectBestEdge()
			if bestIdx == -1 {
				return path
			}
			edge := &curr.Edges[bestIdx]
			gs.ApplyMoveIdx(edge.Move.ToIndex())
			path = append(path, PathStep{Node: edge.Dest, EdgeIdx: bestIdx, PlayerID: gs.PlayerID})
			curr = edge.Dest
		}
	}
}

func (m *MCTSPlayer) expand(curr *MCGSNode, gs *GameState, move Move, playerID int) (*MCGSNode, bool, int) {
	gs.ApplyMove(move)

	// Skip TT lookup during search to save time (low hit rate).
	// We still store the node so it can be found if it becomes the root later.
	child := NewMCGSNode(*gs)
	tt.Store(gs.Hash, child)

	edgeIdx := curr.AddEdge(move, child, playerID)
	return child, true, edgeIdx
}
func (m *MCTSPlayer) Backprop(path []PathStep, result [3]float32) {
	for i := len(path) - 1; i >= 0; i-- {
		step := path[i]
		node := step.Node
		node.UpdateStats(result)

		if i > 0 && step.EdgeIdx != -1 {
			parentStep := path[i-1]
			parentStep.Node.SyncEdge(step.EdgeIdx, node, parentStep.PlayerID)
		}
	}
}

type MCGSEdge struct {
	Move Move
	Dest *MCGSNode
	N    int32
}

const InlineEdgeCap = 4

type MCGSNode struct {
	N            int
	Q            [3]float32
	Edges        []MCGSEdge
	EdgeQs       []float32
	EdgeUs       []float32
	untriedMoves Bitboard
	UCB1Coeff    float32

	edgesBuf [InlineEdgeCap]MCGSEdge
	qsBuf    [InlineEdgeCap]float32
	usBuf    [InlineEdgeCap]float32
}

func (n *MCGSNode) AddEdge(move Move, dest *MCGSNode, playerID int) int {
	idx := len(n.Edges)
	n.Edges = append(n.Edges, MCGSEdge{
		Move: move,
		Dest: dest,
		N:    0,
	})
	n.EdgeQs = append(n.EdgeQs, dest.Q[playerID])
	n.EdgeUs = append(n.EdgeUs, invSqrtTable[1])
	return idx
}

func (n *MCGSNode) selectBestEdge() int {
	if len(n.Edges) == 0 {
		return -1
	}

	if len(n.Edges) >= 8 {
		return selectBestEdgeAVX2(n.EdgeQs, n.EdgeUs, n.UCB1Coeff)
	}

	bestIdx := -1
	bestScore := float32(negInf)
	coeff := n.UCB1Coeff

	for i := range n.Edges {
		score := n.EdgeQs[i] + coeff*n.EdgeUs[i]
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	return bestIdx
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
		n.UCB1Coeff = float32(math.Sqrt(2.0 * math.Log(float64(nPlus1))))
	}
}

func (n *MCGSNode) SyncEdge(idx int, child *MCGSNode, playerID int) {
	edge := &n.Edges[idx]
	edge.N++
	n.EdgeQs[idx] = child.Q[playerID]
	vPlus1 := int(edge.N) + 1
	if vPlus1 < len(invSqrtTable) {
		n.EdgeUs[idx] = invSqrtTable[vPlus1]
	} else {
		n.EdgeUs[idx] = float32(1.0 / math.Sqrt(float64(vPlus1)))
	}
}

func (n *MCGSNode) PopUntriedMove() (Move, bool) {
	moveIdx := PickRandomBit(n.untriedMoves)
	if moveIdx == -1 {
		return Move{}, false
	}
	n.untriedMoves &= ^(Bitboard(1) << uint(moveIdx))
	return MoveFromIndex(moveIdx), true
}

func NewMCGSNode(gs GameState) *MCGSNode {
	_, terminal := gs.IsTerminal()
	var untried Bitboard
	if !terminal {
		untried = gs.GetBestMoves()
	}
	n := &MCGSNode{
		untriedMoves: untried,
	}
	n.Edges = n.edgesBuf[:0]
	n.EdgeQs = n.qsBuf[:0]
	n.EdgeUs = n.usBuf[:0]
	return n
}

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

func ScoreTerminal(activeMask uint8, winnerID int) [3]float32 {
	if winnerID != -1 {
		return ScoreWin(winnerID)
	}
	return ScoreDraw(activeMask)
}

// --- Simulation Logic ---
func RunSimulation(gs *GameState) ([3]float32, int, Board) {
	steps := 0
	for {
		steps++
		winnerID, ok := gs.IsTerminal()
		if ok {
			return ScoreTerminal(gs.ActiveMask, winnerID), steps, gs.Board
		}

		moves := gs.GetBestMoves()
		idx := PickRandomBit(moves)
		if idx == -1 {
			return ScoreDraw(gs.ActiveMask), steps, gs.Board
		}

		gs.ApplyMoveIdx(idx)
	}
}