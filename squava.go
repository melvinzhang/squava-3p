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
func CheckWin(bb Bitboard) bool {
	// Horizontal
	h := bb & (bb >> 1) & Bitboard(^FileH)
	if (h & (h >> 2) & Bitboard(^(FileH | (FileH >> 1)))) != 0 {
		return true
	}
	// Vertical
	v := bb & (bb >> 8)
	if (v & (v >> 16)) != 0 {
		return true
	}
	// Diagonal (A1 -> H8, +9)
	d1 := bb & (bb >> 9) & Bitboard(^FileH)
	if (d1 & (d1 >> 18) & Bitboard(^(FileH | (FileH >> 9)))) != 0 {
		return true
	}
	// Anti-Diagonal (H1 -> A8, +7)
	d2 := bb & (bb >> 7) & Bitboard(^FileA)
	if (d2 & (d2 >> 14) & Bitboard(^(FileA | (FileA >> 7)))) != 0 {
		return true
	}
	return false
}
func CheckLose(bb Bitboard) bool {
	// Horizontal
	h := bb & (bb >> 1) & Bitboard(^FileH)
	if (h & (h >> 1) & Bitboard(^FileH)) != 0 {
		return true
	}
	// Vertical
	v := bb & (bb >> 8)
	if (v & (v >> 8)) != 0 {
		return true
	}
	// Diagonal (+9)
	d1 := bb & (bb >> 9) & Bitboard(^FileH)
	if (d1 & (d1 >> 9) & Bitboard(^FileH)) != 0 {
		return true
	}
	// Anti-Diagonal (+7)
	d2 := bb & (bb >> 7) & Bitboard(^FileA)
	if (d2 & (d2 >> 7) & Bitboard(^FileA)) != 0 {
		return true
	}
	return false
}
func GetWinningMoves(board Board, pID int) Bitboard {
	w, _ := GetWinsAndLoses(board.GetPlayerBoard(pID), ^board.Occupied)
	return w
}
func GetWinsAndLoses(bb Bitboard, empty Bitboard) (wins Bitboard, loses Bitboard) {
	b := uint64(bb)
	e := uint64(empty)
	var w, l uint64
	for d := 0; d < 4; d++ {
		s := shifts[d]
		ml := masksL[d]
		mr := masksR[d]
		l1 := (b << s) & ml
		r1 := (b >> s) & mr
		l2 := (l1 << s) & ml
		r2 := (r1 >> s) & mr
		// Length 3 (loses)
		l |= e & r1 & r2
		l |= e & l1 & r1
		l |= e & l2 & l1
		// Length 4 (wins)
		l3 := (l2 << s) & ml
		r3 := (r2 >> s) & mr
		w |= e & r1 & r2 & r3
		w |= e & l1 & r1 & r2
		w |= e & l2 & l1 & r1
		w |= e & l3 & l2 & l1
	}
	return Bitboard(w), Bitboard(l)
}
func GetValidMoves(board Board, currentPID, nextPID int) Bitboard {
	threats := GetWinningMoves(board, nextPID)
	myWins := GetWinningMoves(board, currentPID)
	return threats | myWins
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

func init() {
	initZobrist()
	for i := 1; i < len(invSqrtTable); i++ {
		invSqrtTable[i] = 1.0 / math.Sqrt(float64(i))
	}
	for i := 1; i < len(coeffTable); i++ {
		coeffTable[i] = 2.0 * math.Sqrt(math.Log(float64(i)))
	}
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

func (m *MCTSPlayer) GetMoveWithContext(board Board, forcedMoves Bitboard, players []int, turnIdx int) Move {
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
	if forcedMoves != 0 {
		root.untriedMoves = forcedMoves
	}
	for root.N < m.iterations {
		path := m.Select(root)
		leaf := path[len(path)-1].Node
		// Expansion
		var result [3]float64
		if leaf.untriedMoves != 0 {
			count := bits.OnesCount64(uint64(leaf.untriedMoves))
			pick := rand.Intn(count)
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
		if forcedMoves != 0 {
			idx := bits.TrailingZeros64(uint64(forcedMoves))
			return MoveFromIndex(idx)
		}
		empty := ^board.Occupied
		if empty != 0 {
			idx := bits.TrailingZeros64(uint64(empty))
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
		for i := range current.Edges {
			edge := &current.Edges[i]
			var u float64
			vPlus1 := edge.Visits + 1
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
		bestEdge := &current.Edges[bestEdgeIdx]
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
func getNextPlayer(currentID int, activeMask uint8) int {
	for i := 1; i <= 2; i++ {
		next := (currentID + i) % 3
		if (activeMask & (1 << uint(next))) != 0 {
			return next
		}
	}
	return -1
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
	empty := ^n.board.Occupied
	myBB := n.board.GetPlayerBoard(n.playerToMoveID)
	nextBB := n.board.GetPlayerBoard(nextID)
	myWins, myLoses := GetWinsAndLoses(myBB, empty)
	nextWins, _ := GetWinsAndLoses(nextBB, empty)
	forced := nextWins | myWins
	if forced != 0 {
		return forced
	}
	targets := (empty & ^myLoses) | myWins
	if targets == 0 {
		targets = empty & myLoses & ^myWins
	}
	if targets == 0 {
		targets = empty
	}
	return targets
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
	if CheckWin(newBoard.GetPlayerBoard(currentID)) {
		return State{board: newBoard, nextPlayerID: -1, activeMask: activeMask, winnerID: currentID}
	}
	if CheckLose(newBoard.GetPlayerBoard(currentID)) {
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
	// Initial threat state
	var wins [3]Bitboard
	empty := ^simBoard.Occupied
	for pID := 0; pID < 3; pID++ {
		if (simMask & (1 << uint(pID))) != 0 {
			wins[pID], _ = GetWinsAndLoses(simBoard.GetPlayerBoard(pID), empty)
		}
	}
	curr := currentID
	for {
		if bits.OnesCount8(simMask) == 1 {
			var res [3]float64
			res[bits.TrailingZeros8(simMask)] = 1.0
			return res
		}
		nextP := getNextPlayer(curr, simMask)
		myBB := simBoard.GetPlayerBoard(curr)
		myWins, myLoses := GetWinsAndLoses(myBB, empty)
		wins[curr] = myWins // Update my threats since I might have new ones
		threats := wins[nextP]
		var movesBitboard Bitboard
		if threats != 0 || myWins != 0 {
			candidates := threats | myWins
			safeCandidates := (candidates & ^myLoses) | (candidates & myWins)
			if safeCandidates != 0 {
				movesBitboard = safeCandidates
			} else {
				movesBitboard = candidates
			}
		} else {
			safeMoves := (empty & ^myLoses) | myWins
			if safeMoves != 0 {
				movesBitboard = safeMoves
			} else {
				movesBitboard = empty & myLoses & ^myWins
			}
		}
		if movesBitboard == 0 {
			return [3]float64{} // Draw
		}
		count := bits.OnesCount64(uint64(movesBitboard))
		if count == 0 {
			return [3]float64{}
		}
		pick := rand.Intn(count)
		var selectedIdx int
		temp := uint64(movesBitboard)
		for i := 0; i < pick; i++ {
			temp &= temp - 1
		}
		selectedIdx = bits.TrailingZeros64(temp)
		simBoard.Set(selectedIdx, curr)
		mask := Bitboard(1) << selectedIdx
		isWin := (myWins & mask) != 0
		isLose := (myLoses & mask) != 0
		empty &= ^mask
		// Incremental update: Clear the blocked square from everyone's threats
		for i := range wins {
			wins[i] &= ^mask
		}
		if isWin {
			var res [3]float64
			res[curr] = 1.0
			return res
		}
		if isLose {
			simMask &= ^(1 << uint(curr))
			if bits.OnesCount8(simMask) == 1 {
				var res [3]float64
				res[bits.TrailingZeros8(simMask)] = 1.0
				return res
			}
			curr = getNextPlayer(curr, simMask)
		} else {
			curr = getNextPlayer(curr, simMask)
		}
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
		forcedMoves := GetValidMoves(g.board, currentPlayer.ID(), nextPlayer.ID())
		var move Move
		if mcts, ok := currentPlayer.(*MCTSPlayer); ok {
			fmt.Printf("%s is thinking...\n", currentPlayer.Name())
			activeIDs := []int{}
			for _, p := range g.players {
				activeIDs = append(activeIDs, p.ID())
			}
			move = mcts.GetMoveWithContext(g.board, forcedMoves, activeIDs, g.turnIdx)
			fmt.Printf("%s chooses %c%d\n", currentPlayer.Name(), move.c+65, move.r+1)
		} else {
			g.PrintBoard()
			move = currentPlayer.GetMove(g.board, forcedMoves)
		}
		g.board.Set(move.ToIndex(), currentPlayer.ID())
		if CheckWin(g.board.GetPlayerBoard(currentPlayer.ID())) {
			g.PrintBoard()
			fmt.Printf("!!! %s wins with 4 in a row! !!!\n", currentPlayer.Name())
			return
		}
		if CheckLose(g.board.GetPlayerBoard(currentPlayer.ID())) {
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
	rand.Seed(time.Now().UnixNano())
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
