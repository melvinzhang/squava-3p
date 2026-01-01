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
)

var (
	masksL1 = [4]uint64{0xFEFEFEFEFEFEFEFE, 0xFFFFFFFFFFFFFFFF, 0xFEFEFEFEFEFEFEFE, 0x7F7F7F7F7F7F7F7F}
	masksR1 = [4]uint64{0x7F7F7F7F7F7F7F7F, 0xFFFFFFFFFFFFFFFF, 0x7F7F7F7F7F7F7F7F, 0xFEFEFEFEFEFEFEFE}
	masksL2 = [4]uint64{0xFCFCFCFCFCFCFCFC, 0xFFFFFFFFFFFFFFFF, 0xFCFCFCFCFCFCFCFC, 0x3F3F3F3F3F3F3F3F}
	masksR2 = [4]uint64{0x3F3F3F3F3F3F3F3F, 0xFFFFFFFFFFFFFFFF, 0x3F3F3F3F3F3F3F3F, 0xFCFCFCFCFCFCFCFC}
	masksL3 = [4]uint64{0xF8F8F8F8F8F8F8F8, 0xFFFFFFFFFFFFFFFF, 0xF8F8F8F8F8F8F8F8, 0x1F1F1F1F1F1F1F1F}
	masksR3 = [4]uint64{0x1F1F1F1F1F1F1F1F, 0xFFFFFFFFFFFFFFFF, 0x1F1F1F1F1F1F1F1F, 0xF8F8F8F8F8F8F8F8}
)

// Board represents the game state using bitboards
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
	mask := Bitboard(uint64(1) << idx)
	b.P[pID] |= mask
	b.Occupied |= mask
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

func GetWinsAndLosses(bb Bitboard, empty Bitboard) (wins Bitboard, loses Bitboard) {
	b := uint64(bb)
	e := uint64(empty)
	var w, l uint64

	// Direction 0: Horizontal (s=1)
	{
		r1 := (b >> 1) & masksR1[0]
		l1 := (b << 1) & masksL1[0]
		r2 := (b >> 2) & masksR2[0]
		l2 := (b << 2) & masksL2[0]

		r1r2 := r1 & r2
		l1l2 := l1 & l2
		l |= e & (r1r2 | r1&l1 | l1l2)

		r3 := (b >> 3) & masksR3[0]
		l3 := (b << 3) & masksL3[0]
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
		r1 := (b >> 9) & masksR1[2]
		l1 := (b << 9) & masksL1[2]
		r2 := (b >> 18) & masksR2[2]
		l2 := (b << 18) & masksL2[2]

		r1r2 := r1 & r2
		l1l2 := l1 & l2
		l |= e & (r1r2 | r1&l1 | l1l2)

		r3 := (b >> 27) & masksR3[2]
		l3 := (b << 27) & masksL3[2]
		w |= e & (r1r2&(r3|l1) | l1l2&(r1|l3))
	}

	// Direction 3: Anti-diagonal (s=7)
	{
		r1 := (b >> 7) & masksR1[3]
		l1 := (b << 7) & masksL1[3]
		r2 := (b >> 14) & masksR2[3]
		l2 := (b << 14) & masksL2[3]

		r1r2 := r1 & r2
		l1l2 := l1 & l2
		l |= e & (r1r2 | r1&l1 | l1l2)

		r3 := (b >> 21) & masksR3[3]
		l3 := (b << 21) & masksL3[3]
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

var (
	invSqrtTable    [100000]float64
	coeffTable      [100000]float64
	tt              []TTEntry
	select8         [256][8]uint8
	nextPlayerTable [3][256]int8
)

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
		invSqrtTable[i] = 1.0 / math.Sqrt(float64(i))
	}
	for i := 1; i < len(coeffTable); i++ {
		coeffTable[i] = 2.0 * math.Sqrt(math.Log(float64(i)))
	}
	tt = make([]TTEntry, TTSize)
}

func getNextPlayer(currentID int, activeMask uint8) int {
	return int(nextPlayerTable[currentID][activeMask])
}

// --- MCTS Player ---
const TTSize = 1 << 24 // ~16M entries
const TTMask = TTSize - 1

type TTEntry struct {
	hash uint64
	node *MCGSNode
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

func (m *MCTSPlayer) GetMove(board Board, players []int, turnIdx int) Move {
	activeMask := uint8(0)
	for _, pID := range players {
		activeMask |= 1 << uint(pID)
	}
	rootHash := ZobristHash(board, players[turnIdx], activeMask)
	idx := int(rootHash & TTMask)
	var root *MCGSNode
	if entry := tt[idx]; entry.hash == rootHash && entry.node != nil {
		if entry.node.Matches(board, players[turnIdx], activeMask, rootHash) {
			root = entry.node
		}
	}

	if root == nil {
		root = NewMCGSNode(board, players[turnIdx], activeMask, rootHash, -1)
		tt[idx] = TTEntry{hash: rootHash, node: root}
	}
	m.root = root

	startRollouts := root.N
	startTime := time.Now()
	totalSteps := 0

	for root.N < m.iterations {
		path := m.Select(root)
		leaf := path[len(path)-1].Node
		// Expansion
		var result [3]float64
		if leaf.untriedMoves != 0 {
			count := bits.OnesCount64(uint64(leaf.untriedMoves))
			var idx int
			if count == 1 {
				idx = bits.TrailingZeros64(uint64(leaf.untriedMoves))
			} else {
				hi, _ := bits.Mul64(xrand(), uint64(count))
				idx = SelectBit64(uint64(leaf.untriedMoves), int(hi))
			}
			move := MoveFromIndex(idx)
			// Remove move from untried
			leaf.untriedMoves &= Bitboard(^(uint64(1) << idx))
			// Calc next state
			state := SimulateStep(leaf.board, leaf.activeMask, leaf.playerToMoveID, move, leaf.hash)
			hash := state.hash
			ttIdx := int(hash & TTMask)
			var nextNode *MCGSNode
			if entry := tt[ttIdx]; entry.hash == hash && entry.node != nil {
				if entry.node.Matches(state.board, state.nextPlayerID, state.activeMask, hash) {
					nextNode = entry.node
				}
			}

			if nextNode != nil {
				result = nextNode.Q // Use existing node's Q for backprop
			} else {
				nextNode = NewMCGSNode(state.board, state.nextPlayerID, state.activeMask, hash, state.winnerID)
				tt[ttIdx] = TTEntry{hash: hash, node: nextNode}

				// Rollout ONLY for new nodes
				if nextNode.winnerID != -1 {
					result[nextNode.winnerID] = 1.0
				} else {
					var s int
					result, s, _ = RunSimulation(nextNode.board, nextNode.activeMask, nextNode.playerToMoveID)
					totalSteps += s
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
	if m.Verbose && elapsed.Seconds() > 0 {
		sps := float64(totalSteps) / elapsed.Seconds()
		fmt.Printf("Rollouts: %d, Steps: %d, Time: %v, SPS: %.2f\n", totalRollouts, totalSteps, elapsed, sps)
	}

	// Stats and Selection
	myID := players[turnIdx]
	if m.Verbose {
		fmt.Printf("Estimated Winrate: %.2f%%\n", root.Q[myID]*100)
	}
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
	if m.Verbose {
		fmt.Println("Top moves:")
		limit := 5
		if len(stats) < limit {
			limit = len(stats)
		}
		for i := 0; i < limit; i++ {
			s := stats[i]
			fmt.Printf("  %c%d: Visits: %d, Winrate: %.2f%%\n", s.mv.c+65, s.mv.r+1, s.visits, s.winrate*100)
		}
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
		invN := 1.0 / float64(node.N)
		node.Q[0] += (result[0] - node.Q[0]) * invN
		node.Q[1] += (result[1] - node.Q[1]) * invN
		node.Q[2] += (result[2] - node.Q[2]) * invN

		// Update cached coeff
		nPlus1 := node.N + 1
		if nPlus1 < len(coeffTable) {
			node.UCB1Coeff = coeffTable[nPlus1]
		} else {
			node.UCB1Coeff = 2.0 * math.Sqrt(math.Log(float64(nPlus1)))
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
	hash           uint64
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

func NewMCGSNode(board Board, playerToMoveID int, activeMask uint8, hash uint64, winnerID int) *MCGSNode {
	var untried Bitboard
	if winnerID == -1 {
		nextP := getNextPlayer(playerToMoveID, activeMask)
		untried = GetBestMoves(board, AnalyzeThreats(board, playerToMoveID, nextP))
	}
	n := &MCGSNode{
		board:          board,
		hash:           hash,
		playerToMoveID: playerToMoveID,
		activeMask:     activeMask,
		winnerID:       winnerID,
		untriedMoves:   untried,
	}
	return n
}
func (n *MCGSNode) Matches(board Board, playerToMoveID int, activeMask uint8, hash uint64) bool {
	return n.hash == hash && n.playerToMoveID == playerToMoveID && n.activeMask == activeMask && n.board == board
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
	hash         uint64
}

// --- Simulation Logic ---
func SimulateStep(board Board, activeMask uint8, currentID int, move Move, currentHash uint64) State {
	newBoard := board
	idx := move.ToIndex()
	mask := Bitboard(uint64(1) << idx)
	newBoard.P[currentID] |= mask
	newBoard.Occupied |= mask

	newHash := currentHash ^ zobristP[currentID][idx] ^ zobristTurn[currentID]

	isWin, isLoss := CheckBoard(newBoard.GetPlayerBoard(currentID))
	if isWin {
		return State{board: newBoard, nextPlayerID: -1, activeMask: activeMask, winnerID: currentID, hash: newHash}
	}
	if isLoss {
		newMask := activeMask & ^(1 << uint(currentID))
		if bits.OnesCount8(newMask) == 1 {
			winnerID := bits.TrailingZeros8(newMask)
			newHash ^= zobristActive[activeMask] ^ zobristActive[newMask]
			return State{board: newBoard, nextPlayerID: -1, activeMask: newMask, winnerID: winnerID, hash: newHash}
		}
		nextID := getNextPlayer(currentID, newMask)
		newHash ^= zobristTurn[nextID] ^ zobristActive[activeMask] ^ zobristActive[newMask]
		return State{board: newBoard, nextPlayerID: nextID, activeMask: newMask, winnerID: -1, hash: newHash}
	}
	nextID := getNextPlayer(currentID, activeMask)
	newHash ^= zobristTurn[nextID]
	return State{board: newBoard, nextPlayerID: nextID, activeMask: activeMask, winnerID: -1, hash: newHash}
}
func pdep(src, mask uint64) uint64

func SelectBit64(v uint64, k int) int {
	return bits.TrailingZeros64(pdep(uint64(1)<<uint(k), v))
}

func RunSimulation(board Board, activeMask uint8, currentID int) ([3]float64, int, Board) {
	simBoard := board
	simMask := activeMask
	curr := currentID
	steps := 0

	empty := ^simBoard.Occupied
	var allWins, allLoses [3]Bitboard

	// Initial threat analysis
	for p := 0; p < 3; p++ {
		if (simMask & (1 << uint(p))) != 0 {
			allWins[p], allLoses[p] = GetWinsAndLosses(simBoard.P[p], empty)
		}
	}

	for {
		steps++
		if simMask&(simMask-1) == 0 {
			var res [3]float64
			res[bits.TrailingZeros8(simMask)] = 1.0
			return res, steps, simBoard
		}

		myWins := allWins[curr]
		if myWins != 0 {
			var res [3]float64
			res[curr] = 1.0
			return res, steps, simBoard
		}

		nextP := int(nextPlayerTable[curr][simMask])
		nextWins := allWins[nextP]
		myLoses := allLoses[curr]

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
			return [3]float64{}, steps, simBoard
		}

		var selectedIdx int
		count := bits.OnesCount64(uint64(moves))
		if count == 1 {
			selectedIdx = bits.TrailingZeros64(uint64(moves))
		} else {
			hi, _ := bits.Mul64(xrand(), uint64(count))
			selectedIdx = SelectBit64(uint64(moves), int(hi))
		}

		mask := Bitboard(uint64(1) << selectedIdx)
		simBoard.Occupied |= mask
		simBoard.P[curr] |= mask
		empty &= ^mask

		if mustCheckLoss && (allLoses[curr]&mask) != 0 {
			simMask &= ^(1 << uint(curr))
			if simMask&(simMask-1) == 0 {
				var res [3]float64
				res[bits.TrailingZeros8(simMask)] = 1.0
				return res, steps, simBoard
			}
			curr = int(nextPlayerTable[curr][simMask])
			// Update others and skip recalculation for eliminated curr
			for p := 0; p < 3; p++ {
				if (simMask & (1 << uint(p))) != 0 {
					allWins[p] &= ^mask
					allLoses[p] &= ^mask
				}
			}
			continue
		}

		// Update others
		for p := 0; p < 3; p++ {
			if p != curr && (simMask&(1<<uint(p))) != 0 {
				allWins[p] &= ^mask
				allLoses[p] &= ^mask
			}
		}
		// Recalculate for curr
		allWins[curr], allLoses[curr] = GetWinsAndLosses(simBoard.P[curr], empty)
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
	fmt.Println("Board Size: 8x8")
	fmt.Println("Rules: 4-in-a-row wins. 3-in-a-row loses.")
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
		fmt.Printf("Turn: %s (%s)\n", currentPlayer.Name(), currentPlayer.Symbol())

		activeIDs := make([]int, len(g.players))
		for i, p := range g.players {
			activeIDs[i] = p.ID()
		}

		if _, ok := currentPlayer.(*MCTSPlayer); ok {
			fmt.Printf("%s is thinking...\n", currentPlayer.Name())
		}

		move := currentPlayer.GetMove(g.board, activeIDs, g.turnIdx)

		if _, ok := currentPlayer.(*MCTSPlayer); ok {
			fmt.Printf("%s chooses %c%d\n", currentPlayer.Name(), move.c+65, move.r+1)
		}
		g.board.Set(move.ToIndex(), currentPlayer.ID())
		isWin, isLoss := CheckBoard(g.board.GetPlayerBoard(currentPlayer.ID()))
		if isWin {
			g.PrintBoard()
			fmt.Printf("!!! %s wins with 4 in a row! !!!\n", currentPlayer.Name())
			return
		}
		if isLoss {
			g.PrintBoard()
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
