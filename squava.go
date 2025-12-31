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
	"weak"
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
	// This is biased, but fast. For MCTS rollouts, it's an acceptable trade-off.
	return int(xrand() % uint64(n))
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
)

var (
	shifts = [4]uint{1, 8, 9, 7}
	masksL = [4]uint64{0xFEFEFEFEFEFEFEFE, 0xFFFFFFFFFFFFFFFF, 0xFEFEFEFEFEFEFEFE, 0x7F7F7F7F7F7F7F7F}
	masksR = [4]uint64{0x7F7F7F7F7F7F7F7F, 0xFFFFFFFFFFFFFFFF, 0x7F7F7F7F7F7F7F7F, 0xFEFEFEFEFEFEFEFE}
)

// Board represents the game state using bitboards
type Board struct {
	P1, P2, P3 Bitboard
	Occupied   Bitboard
}
type Bitboard uint64
type Player interface {
	GetMove(board Board, forcedMoves Bitboard) Move
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
	r, c int
}

func (m Move) ToIndex() int {
	return m.r*8 + m.c
}
func MoveFromIndex(idx int) Move {
	return Move{r: idx / 8, c: idx % 8}
}

// --- Bitboard Logic ---
func (b *Board) Set(idx int, pID int) {
	mask := uint64(1) << idx
	b.Occupied |= Bitboard(mask)
	switch pID {
	case 0:
		b.P1 |= Bitboard(mask)
	case 1:
		b.P2 |= Bitboard(mask)
	case 2:
		b.P3 |= Bitboard(mask)
	}
}
func (b *Board) GetPlayerBoard(pID int) Bitboard {
	switch pID {
	case 0:
		return b.P1
	case 1:
		return b.P2
	case 2:
		return b.P3
	}
	return 0
}
func CheckBoard(bb Bitboard) (isWin, isLoss bool) {
	b := uint64(bb)
	var h, v, d1, d2 uint64

	// Horizontal
	h = b & (b >> 1)
	// Vertical
	v = b & (b >> 8)
	// Diagonal
	d1 = b & (b >> 9)
	// Anti-diagonal
	d2 = b & (b >> 7)

	// A win is 4-in-a-row, which implies two adjacent pairs.
	// A loss is 3-in-a-row, which implies one pair and an adjacent piece.
	isWin = ((h & (h >> 2)) & 0xFCFCFCFCFCFCFCFC) != 0 || // H
		(v & (v >> 16)) != 0 || // V
		((d1 & (d1 >> 18)) & 0xFCFCFCFCFCFCFCFC) != 0 || // D1
				((d2 & (d2 >> 14)) & 0x7F7F7F7F7F7F7F7F) != 0 // D2
		
			isLoss = !isWin && (((h & (h >> 1)) & 0xFDFDFDFDFDFDFDFD) != 0 || // H
				(v & (v >> 8)) != 0 || // V
				((d1 & (d1 >> 9)) & 0xFEFEFEFEFEFEFEFE) != 0 || // D1
				((d2 & (d2 >> 7)) & 0x7F7F7F7F7F7F7F7F) != 0) // D2
		
			return
		}
		
		func GetWinsAndLoses(bb Bitboard, empty Bitboard) (wins Bitboard, loses Bitboard) {
	b := uint64(bb)
	e := uint64(empty)
	var w, l uint64

	// Direction 0: Horizontal (s=1)
	{
		ml, mr := masksL[0], masksR[0]
		l1 := (b << 1) & ml
		r1 := (b >> 1) & mr
		l2 := (l1 << 1) & ml
		r2 := (r1 >> 1) & mr
		A := r1 & r2
		B := l1 & r1
		C := l1 & l2
		l |= e & (A | B | C)
		w |= e & (A&(r2>>1&mr) | A&l1 | C&r1 | C&(l2<<1&ml))
	}

	// Direction 1: Vertical (s=8)
	{
		l1 := (b << 8)
		r1 := (b >> 8)
		l2 := (l1 << 8)
		r2 := (r1 >> 8)
		A := r1 & r2
		B := l1 & r1
		C := l1 & l2
		l |= e & (A | B | C)
		w |= e & (A&(r2>>8) | A&l1 | C&r1 | C&(l2<<8))
	}

	// Direction 2: Diagonal (s=9)
	{
		ml, mr := masksL[2], masksR[2]
		l1 := (b << 9) & ml
		r1 := (b >> 9) & mr
		l2 := (l1 << 9) & ml
		r2 := (r1 >> 9) & mr
		A := r1 & r2
		B := l1 & r1
		C := l1 & l2
		l |= e & (A | B | C)
		w |= e & (A&(r2>>9&mr) | A&l1 | C&r1 | C&(l2<<9&ml))
	}

	// Direction 3: Anti-diagonal (s=7)
	{
		ml, mr := masksL[3], masksR[3]
		l1 := (b << 7) & ml
		r1 := (b >> 7) & mr
		l2 := (l1 << 7) & ml
		r2 := (r1 >> 7) & mr
		A := r1 & r2
		B := l1 & r1
		C := l1 & l2
		l |= e & (A | B | C)
		w |= e & (A&(r2>>7&mr) | A&l1 | C&r1 | C&(l2<<7&ml))
	}

	return Bitboard(w), Bitboard(l)
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
	myWins, myLoses := GetWinsAndLoses(board.GetPlayerBoard(currentPID), empty)
	nextWins, _ := GetWinsAndLoses(board.GetPlayerBoard(nextPID), empty)
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
func (h *HumanPlayer) GetMove(board Board, forcedMoves Bitboard) Move {
	reader := bufio.NewReader(os.Stdin)
	for {
		prompt := fmt.Sprintf("%s (%s), enter your move (e.g., A1): ", h.info.name, h.info.symbol)
		if forcedMoves != 0 {
			forcedStr := []string{}
			temp := forcedMoves
			for temp != 0 {
				idx := bits.TrailingZeros64(uint64(temp))
				m := MoveFromIndex(idx)
				forcedStr = append(forcedStr, fmt.Sprintf("%c%d", m.c+65, m.r+1))
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
		move := Move{r, c}
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

var invSqrtTable [1000001]float64
var coeffTable [1000001]float64

var nextPlayerTable [3][256]int8

func init() {
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
		invSqrtTable[i] = 1.0 / math.Sqrt(float64(i))
	}
	for i := 1; i < len(coeffTable); i++ {
		coeffTable[i] = 2.0 * math.Sqrt(math.Log(float64(i)))
	}
}

func getNextPlayer(currentID int, activeMask uint8) int {
	return int(nextPlayerTable[currentID][activeMask])
}

// --- MCTS Player ---
const TTSize = 1 << 20 // 1M entries
const TTMask = TTSize - 1

type TTEntry struct {
	hash uint64
	ptr  weak.Pointer[MCGSNode]
}

type MCTSPlayer struct {
	info       PlayerInfo
	iterations int
	tt         []TTEntry
	path       []PathStep
}

func NewMCTSPlayer(name, symbol string, id int, iterations int) *MCTSPlayer {
	return &MCTSPlayer{
		info:       PlayerInfo{name: name, symbol: symbol, id: id},
		iterations: iterations,
		tt:         make([]TTEntry, TTSize),
		path:       make([]PathStep, 0, 64),
	}
}
func (m *MCTSPlayer) Name() string   { return m.info.name }
func (m *MCTSPlayer) Symbol() string { return m.info.symbol }
func (m *MCTSPlayer) ID() int        { return m.info.id }
func (m *MCTSPlayer) GetMove(board Board, forcedMoves Bitboard) Move {
	return Move{0, 0}
}
func ZobristHash(board Board, playerToMoveID int, activeMask uint8) uint64 {
	var h uint64
	if playerToMoveID >= 0 && playerToMoveID < 3 {
		h = zobristTurn[playerToMoveID]
	}
	h ^= zobristActive[activeMask]
	p1 := uint64(board.P1)
	for p1 != 0 {
		idx := bits.TrailingZeros64(p1)
		h ^= zobristP[0][idx]
		p1 &= p1 - 1
	}
	p2 := uint64(board.P2)
	for p2 != 0 {
		idx := bits.TrailingZeros64(p2)
		h ^= zobristP[1][idx]
		p2 &= p2 - 1
	}
	p3 := uint64(board.P3)
	for p3 != 0 {
		idx := bits.TrailingZeros64(p3)
		h ^= zobristP[2][idx]
		p3 &= p3 - 1
	}
	return h
}

func (m *MCTSPlayer) GetMoveWithContext(board Board, players []int, turnIdx int) Move {
	if m.tt == nil {
		m.tt = make([]TTEntry, TTSize)
	}

	activeMask := uint8(0)
	for _, pID := range players {
		activeMask |= 1 << uint(pID)
	}
	rootHash := ZobristHash(board, players[turnIdx], activeMask)
	idx := int(rootHash & TTMask)
	var root *MCGSNode
	if entry := m.tt[idx]; entry.hash == rootHash {
		candidate := entry.ptr.Value()
		if candidate != nil && candidate.Matches(board, players[turnIdx], activeMask) {
			root = candidate
		}
	}

	if root == nil {
		root = NewMCGSNode(board, players[turnIdx], activeMask)
		m.tt[idx] = TTEntry{hash: rootHash, ptr: weak.Make(root)}
	}

	startRollouts := root.N
	startTime := time.Now()

	for root.N < m.iterations {
		path := m.Select(root)
		leaf := path[len(path)-1].Node
		// Expansion
		var result [3]float64
		if leaf.untriedMoves != 0 {
			count := bits.OnesCount64(uint64(leaf.untriedMoves))
			pick := randIntn(count)
			temp := uint64(leaf.untriedMoves)
			for j := 0; j < pick; j++ {
				temp &= temp - 1
			}
			idx := bits.TrailingZeros64(temp)
			move := MoveFromIndex(idx)
			// Remove move from untried
			leaf.untriedMoves &= Bitboard(^(uint64(1) << idx))
			// Calc next state
			state := SimulateStep(leaf.board, leaf.activeMask, leaf.playerToMoveID, move)
			hash := ZobristHash(state.board, state.nextPlayerID, state.activeMask)
			ttIdx := int(hash & TTMask)
			var nextNode *MCGSNode
			if entry := m.tt[ttIdx]; entry.hash == hash {
				candidate := entry.ptr.Value()
				if candidate != nil && candidate.Matches(state.board, state.nextPlayerID, state.activeMask) {
					nextNode = candidate
				}
			}

			if nextNode != nil {
				result = nextNode.Q // Use existing node's Q for backprop
			} else {
				nextNode = NewMCGSNode(state.board, state.nextPlayerID, state.activeMask)
				nextNode.winnerID = state.winnerID
				m.tt[ttIdx] = TTEntry{hash: hash, ptr: weak.Make(nextNode)}

				// Rollout ONLY for new nodes
				if nextNode.winnerID != -1 {
					result[nextNode.winnerID] = 1.0
				} else {
					result = RunSimulation(nextNode.board, nextNode.activeMask, nextNode.playerToMoveID)
				}
				nextNode.U = result
				nextNode.Q = result
			}
			// Add Edge
			edgeIdx := len(leaf.Edges)
			leaf.Edges = append(leaf.Edges, MCGSEdge{
				Move:    move,
				Dest:    nextNode,
				Visits:  0,
				CachedQ: nextNode.Q[leaf.playerToMoveID],
			})
			// Add to path
			path = append(path, PathStep{Node: nextNode, EdgeIdx: edgeIdx})
		} else {
			// Leaf is terminal or fully expanded
			result = leaf.Q
		}
		// Backprop
		m.Backprop(path, result)
	}

	elapsed := time.Since(startTime)
	totalRollouts := root.N - startRollouts
	if elapsed.Seconds() > 0 {
		rolloutsPerSec := float64(totalRollouts) / elapsed.Seconds()
		fmt.Printf("Rollouts: %d, Time: %v, RPS: %.2f\n", totalRollouts, elapsed, rolloutsPerSec)
	}

	// Stats and Selection
	myID := players[turnIdx]
	fmt.Printf("Estimated Winrate: %.2f%%\n", root.Q[myID]*100)
	bestVisits := -1
	var bestMove Move
	// Sort moves by visits for cleaner output
	type MoveStat struct {
		mv      Move
		visits  int
		winrate float64
	}
	stats := []MoveStat{}
	for _, edge := range root.Edges {
		stats = append(stats, MoveStat{edge.Move, edge.Visits, edge.Dest.Q[myID]})
		if edge.Visits > bestVisits {
			bestVisits = edge.Visits
			bestMove = edge.Move
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
	// Print top 5 moves
	fmt.Println("Top moves:")
	limit := 5
	if len(stats) < limit {
		limit = len(stats)
	}
	for i := 0; i < limit; i++ {
		s := stats[i]
		fmt.Printf("  %c%d: Visits: %d, Winrate: %.2f%%\n", s.mv.c+65, s.mv.r+1, s.visits, s.winrate*100)
	}
	if bestVisits == -1 {
		// Fallback
		nextID := getNextPlayer(players[turnIdx], activeMask)
		threats := AnalyzeThreats(board, players[turnIdx], nextID)
		moves := GetBestMoves(board, threats)
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

func (m *MCTSPlayer) Select(root *MCGSNode) []PathStep {
	m.path = m.path[:0]
	m.path = append(m.path, PathStep{Node: root, EdgeIdx: -1})
	current := root
	for {
		if current.untriedMoves != 0 {
			return m.path // Expand here
		}
		if len(current.Edges) == 0 {
			return m.path // Terminal
		}
		bestScore := math.Inf(-1)
		bestEdgeIdx := -1
		c := current.UCB1Coeff
		edges := current.Edges
		for i := range edges {
			edge := &edges[i]
			vPlus1 := edge.Visits + 1
			var u float64
			if vPlus1 < len(invSqrtTable) {
				u = c * invSqrtTable[vPlus1]
			} else {
				u = c / math.Sqrt(float64(vPlus1))
			}
			score := edge.CachedQ + u
			if score > bestScore {
				bestScore = score
				bestEdgeIdx = i
			}
		}
		if bestEdgeIdx == -1 {
			break
		}
		bestEdge := &edges[bestEdgeIdx]
		m.path = append(m.path, PathStep{Node: bestEdge.Dest, EdgeIdx: bestEdgeIdx})
		current = bestEdge.Dest
	}
	return m.path
}
func (m *MCTSPlayer) Backprop(path []PathStep, result [3]float64) {
	for i := len(path) - 1; i >= 0; i-- {
		step := path[i]
		node := step.Node
		node.N++
		fn := float64(node.N)
		node.Q[0] += (result[0] - node.Q[0]) / fn
		node.Q[1] += (result[1] - node.Q[1]) / fn
		node.Q[2] += (result[2] - node.Q[2]) / fn
		// Update cached coeff
		if node.N+1 < len(coeffTable) {
			node.UCB1Coeff = coeffTable[node.N+1]
		} else {
			node.UCB1Coeff = 2.0 * math.Sqrt(math.Log(float64(node.N+1)))
		}
		if i > 0 && step.EdgeIdx != -1 {
			parent := path[i-1].Node
			edge := &parent.Edges[step.EdgeIdx]
			edge.Visits++
			edge.CachedQ = node.Q[parent.playerToMoveID]
		}
	}
}

type MCGSNode struct {
	board          Board
	N              int
	U              [3]float64
	Q              [3]float64
	Edges          []MCGSEdge
	playerToMoveID int
	activeMask     uint8
	winnerID       int
	untriedMoves   Bitboard
	UCB1Coeff      float64
}
type MCGSEdge struct {
	Move    Move
	Dest    *MCGSNode
	Visits  int
	CachedQ float64
}

func NewMCGSNode(board Board, playerToMoveID int, activeMask uint8) *MCGSNode {
	node := &MCGSNode{
		board:          board,
		playerToMoveID: playerToMoveID,
		activeMask:     activeMask,
		winnerID:       -1,
		UCB1Coeff:      0,
	}
	node.untriedMoves = node.GetPossibleMoves()
	return node
}

func (n *MCGSNode) Matches(board Board, playerToMoveID int, activeMask uint8) bool {
	return n.board == board && n.playerToMoveID == playerToMoveID && n.activeMask == activeMask
}

func (n *MCGSNode) GetPossibleMoves() Bitboard {
	if n.winnerID != -1 {
		return 0
	}
	if bits.OnesCount8(n.activeMask) < 2 {
		return 0
	}
	if (n.activeMask & (1 << uint(n.playerToMoveID))) == 0 {
		return 0
	}
	nextID := getNextPlayer(n.playerToMoveID, n.activeMask)
	threats := AnalyzeThreats(n.board, n.playerToMoveID, nextID)
	return GetBestMoves(n.board, threats)
}

type State struct {
	board        Board
	nextPlayerID int
	activeMask   uint8
	winnerID     int
}

// --- Simulation Logic ---
func SimulateStep(board Board, activeMask uint8, currentID int, move Move) State {
	newBoard := board
	newBoard.Set(move.ToIndex(), currentID)
	isWin, isLoss := CheckBoard(newBoard.GetPlayerBoard(currentID))
	if isWin {
		return State{board: newBoard, nextPlayerID: -1, activeMask: activeMask, winnerID: currentID}
	}
	if isLoss {
		newMask := activeMask & ^(1 << uint(currentID))
		if bits.OnesCount8(newMask) == 1 {
			winnerID := bits.TrailingZeros8(newMask)
			return State{board: newBoard, nextPlayerID: -1, activeMask: newMask, winnerID: winnerID}
		}
		nextID := getNextPlayer(currentID, newMask)
		return State{board: newBoard, nextPlayerID: nextID, activeMask: newMask, winnerID: -1}
	}
	nextID := getNextPlayer(currentID, activeMask)
	return State{board: newBoard, nextPlayerID: nextID, activeMask: activeMask, winnerID: -1}
}
func RunSimulation(board Board, activeMask uint8, currentID int) [3]float64 {
	simBoard := board
	simMask := activeMask
	curr := currentID
	for {
		if simMask&(simMask-1) == 0 {
			var res [3]float64
			res[bits.TrailingZeros8(simMask)] = 1.0
			return res
		}

		nextP := getNextPlayer(curr, simMask)
		empty := ^simBoard.Occupied
		myWins, myLoses := GetWinsAndLoses(simBoard.GetPlayerBoard(curr), empty)

		if myWins != 0 {
			var res [3]float64
			res[curr] = 1.0
			return res
		}

		nextWins, _ := GetWinsAndLoses(simBoard.GetPlayerBoard(nextP), empty)

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
			return [3]float64{}
		}

		var selectedIdx int
		count := bits.OnesCount64(uint64(moves))
		if count == 1 {
			selectedIdx = bits.TrailingZeros64(uint64(moves))
		} else {
			pick := randIntn(count)
			temp := uint64(moves)
			for i := 0; i < pick; i++ {
				temp &= temp - 1
			}
			selectedIdx = bits.TrailingZeros64(temp)
		}

		simBoard.Set(selectedIdx, curr)

		if mustCheckLoss {
			_, isLoss := CheckBoard(simBoard.GetPlayerBoard(curr))
			if isLoss {
				simMask &= ^(1 << uint(curr))
				if simMask&(simMask-1) == 0 {
					var res [3]float64
					res[bits.TrailingZeros8(simMask)] = 1.0
					return res
				}
				curr = getNextPlayer(curr, simMask)
				continue
			}
		}

		curr = nextP
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
			mask := uint64(1) << idx
			if (uint64(g.board.P1) & mask) != 0 {
				symbol = "X"
			} else if (uint64(g.board.P2) & mask) != 0 {
				symbol = "O"
			} else if (uint64(g.board.P3) & mask) != 0 {
				symbol = "Z"
			}
			fmt.Printf("%s ", symbol)
		}
		fmt.Println()
	}
}
func (g *SquavaGame) Run() {
	fmt.Println("Starting 3-Player Squava!")
	fmt.Println("Board Size: 8x8")
	fmt.Println("Rules: 4-in-a-row wins. 3-in-a-row loses.")
	for {
		if len(g.players) == 0 {
			fmt.Println("All players eliminated? Draw.")
			break
		}
		if len(g.players) == 1 {
			fmt.Printf("%s wins as the last player standing!\n", g.players[0].Name())
			break
		}
		currentPlayer := g.players[g.turnIdx]
		nextPlayerIdx := (g.turnIdx + 1) % len(g.players)
		nextPlayer := g.players[nextPlayerIdx]

		fmt.Printf("Turn: %s (%s)\n", currentPlayer.Name(), currentPlayer.Symbol())
		var move Move
		if mcts, ok := currentPlayer.(*MCTSPlayer); ok {
			fmt.Printf("%s is thinking...\n", currentPlayer.Name())
			activeIDs := []int{}
			for _, p := range g.players {
				activeIDs = append(activeIDs, p.ID())
			}
			move = mcts.GetMoveWithContext(g.board, activeIDs, g.turnIdx)
			fmt.Printf("%s chooses %c%d\n", currentPlayer.Name(), move.c+65, move.r+1)
		} else {
			g.PrintBoard()
			forcedMoves := GetForcedMoves(g.board, currentPlayer.ID(), nextPlayer.ID())
			move = currentPlayer.GetMove(g.board, forcedMoves)
		}
		g.board.Set(move.ToIndex(), currentPlayer.ID())
		isWin, isLoss := CheckBoard(g.board.GetPlayerBoard(currentPlayer.ID()))
		if isWin {
			g.PrintBoard()
			fmt.Printf("!!! %s wins with 4 in a row! !!!\n", currentPlayer.Name())
			return
		}
		if isLoss {
			fmt.Printf("Oops! %s made 3 in a row and is eliminated!\n", currentPlayer.Name())
			g.players = append(g.players[:g.turnIdx], g.players[g.turnIdx+1:]...)
			if g.turnIdx >= len(g.players) {
				g.turnIdx = 0
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
	xorState = uint64(time.Now().UnixNano())
	if xorState == 0 {
		xorState = 1
	}
	game := NewSquavaGame()
	createPlayer := func(t, name, symbol string, id int) Player {
		if t == "mcts" {
			return NewMCTSPlayer(name, symbol, id, *iterations)
		}
		return NewHumanPlayer(name, symbol, id)
	}
	game.AddPlayer(createPlayer(*p1Type, "Player 1", "X", 0))
	game.AddPlayer(createPlayer(*p2Type, "Player 2", "O", 1))
	game.AddPlayer(createPlayer(*p3Type, "Player 3", "Z", 2))
	game.Run()
}
