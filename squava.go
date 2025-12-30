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

const (
	BoardSize  = 8
)

// Bitboard constants
const (
	FileA uint64 = 0x0101010101010101
	FileH uint64 = 0x8080808080808080
	Full  uint64 = 0xFFFFFFFFFFFFFFFF
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
	if (h & (h >> 2) & Bitboard(^(FileH | (FileH >> 1)))) != 0 { return true }
	
	// Vertical
	v := bb & (bb >> 8)
	if (v & (v >> 16)) != 0 { return true }
	
	// Diagonal (A1 -> H8, +9)
	d1 := bb & (bb >> 9) & Bitboard(^FileH)
	if (d1 & (d1 >> 18) & Bitboard(^(FileH | (FileH>>9)))) != 0 { return true }
	
	// Anti-Diagonal (H1 -> A8, +7)
	d2 := bb & (bb >> 7) & Bitboard(^FileA)
	if (d2 & (d2 >> 14) & Bitboard(^(FileA | (FileA>>7)))) != 0 { return true }
	
	return false
}

func CheckLose(bb Bitboard) bool {
	// Horizontal
	h := bb & (bb >> 1) & Bitboard(^FileH)
	if (h & (h >> 1) & Bitboard(^FileH)) != 0 { return true }
	
	// Vertical
	v := bb & (bb >> 8)
	if (v & (v >> 8)) != 0 { return true }
	
	// Diagonal (+9)
	d1 := bb & (bb >> 9) & Bitboard(^FileH)
	if (d1 & (d1 >> 9) & Bitboard(^FileH)) != 0 { return true }
	
	// Anti-Diagonal (+7)
	d2 := bb & (bb >> 7) & Bitboard(^FileA)
	if (d2 & (d2 >> 7) & Bitboard(^FileA)) != 0 { return true }
	
	return false
}

func GetWinningMoves(board Board, pID int) Bitboard {
	return GetMovesThatComplete(board.GetPlayerBoard(pID), ^board.Occupied, 4)
}

func GetMovesThatComplete(bb Bitboard, empty Bitboard, length int) Bitboard {
	b := uint64(bb)
	e := uint64(empty)
	var moves uint64

	shifts := [4]uint{1, 8, 9, 7}
	masksL := [4]uint64{0xFEFEFEFEFEFEFEFE, 0xFFFFFFFFFFFFFFFF, 0xFEFEFEFEFEFEFEFE, 0x7F7F7F7F7F7F7F7F}
	masksR := [4]uint64{0x7F7F7F7F7F7F7F7F, 0xFFFFFFFFFFFFFFFF, 0x7F7F7F7F7F7F7F7F, 0xFEFEFEFEFEFEFEFE}

	for d := 0; d < 4; d++ {
		s := shifts[d]
		ml := masksL[d]
		mr := masksR[d]

		l1 := (b << s) & ml
		r1 := (b >> s) & mr
		l2 := (l1 << s) & ml
		r2 := (r1 >> s) & mr

		if length == 4 {
			l3 := (l2 << s) & ml
			r3 := (r2 >> s) & mr
			moves |= e & r1 & r2 & r3
			moves |= e & l1 & r1 & r2
			moves |= e & l2 & l1 & r1
			moves |= e & l3 & l2 & l1
		} else {
			moves |= e & r1 & r2
			moves |= e & l1 & r1
			moves |= e & l2 & l1
		}
	}
	return Bitboard(moves)
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
func (h *HumanPlayer) Name() string { return h.info.name }
func (h *HumanPlayer) Symbol() string { return h.info.symbol }
func (h *HumanPlayer) ID() int { return h.info.id }

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

// --- MCTS Player ---

type MCTSPlayer struct {
	info       PlayerInfo
	iterations int
	tt         map[BoardHash]*MCGSNode
}

func NewMCTSPlayer(name, symbol string, id int, iterations int) *MCTSPlayer {
	return &MCTSPlayer{
		info:       PlayerInfo{name: name, symbol: symbol, id: id},
		iterations: iterations,
		tt:         make(map[BoardHash]*MCGSNode),
	}
}
func (m *MCTSPlayer) Name() string { return m.info.name }
func (m *MCTSPlayer) Symbol() string { return m.info.symbol }
func (m *MCTSPlayer) ID() int { return m.info.id }

func (m *MCTSPlayer) GetMove(board Board, forcedMoves Bitboard) Move {
    return Move{0,0}
}

func (m *MCTSPlayer) GetMoveWithContext(board Board, forcedMoves Bitboard, players []int, turnIdx int) Move {
    if m.tt == nil {
        m.tt = make(map[BoardHash]*MCGSNode)
    }

    activeMask := uint8(0)
    for _, pID := range players {
        activeMask |= 1 << pID
    }

    rootHash := BoardHash{board.P1, board.P2, board.P3, players[turnIdx], activeMask}
    root, ok := m.tt[rootHash]
    if !ok {
        root = NewMCGSNode(board, players[turnIdx], players)
        m.tt[rootHash] = root
    }

	if forcedMoves != 0 {
		root.untriedMoves = forcedMoves
	}

	for root.N < m.iterations {
		path := m.Select(root)
        leaf := path[len(path)-1].Node
        
        // Expansion
        var nextNode *MCGSNode
        var move Move
        
        if leaf.untriedMoves != 0 {
            count := bits.OnesCount64(uint64(leaf.untriedMoves))
            pick := rand.Intn(count)
            
            temp := uint64(leaf.untriedMoves)
            for j := 0; j < pick; j++ {
                temp &= temp - 1
            }
            idx := bits.TrailingZeros64(temp)
            move = MoveFromIndex(idx)
            
            // Remove move from untried
            leaf.untriedMoves &= Bitboard(^(uint64(1) << idx))
            
            // Calc next state
            state := SimulateStep(leaf.board, leaf.remainingPlayers, leaf.playerToMoveID, move)
            
            nextActiveMask := uint8(0)
            for _, pID := range state.remainingPlayers {
                nextActiveMask |= 1 << pID
            }
            hash := BoardHash{state.board.P1, state.board.P2, state.board.P3, state.nextPlayerID, nextActiveMask}
            
            if existing, ok := m.tt[hash]; ok {
                nextNode = existing
            } else {
                nextNode = NewMCGSNode(state.board, state.nextPlayerID, state.remainingPlayers)
                nextNode.winnerID = state.winnerID
                m.tt[hash] = nextNode
                
                // Rollout ONLY for new nodes
                var result [3]float64
                if nextNode.winnerID != -1 {
                    result[nextNode.winnerID] = 1.0
                } else {
                    result = RunSimulation(nextNode.board, nextNode.remainingPlayers, nextNode.playerToMoveID)
                }
                
                nextNode.U = result
                nextNode.Q = result
            }
            
            // Add Edge
            if leaf.Edges == nil { leaf.Edges = make(map[Move]*MCGSEdge) }
            if _, ok := leaf.Edges[move]; !ok {
                 leaf.Edges[move] = &MCGSEdge{Dest: nextNode, Visits: 0}
            }
            // Add to path
            path = append(path, PathStep{Node: nextNode, Move: move, Edge: leaf.Edges[move]})
        }
        
        // Backprop
        m.Backprop(path)
	}

    // Stats and Selection
    fmt.Printf("MCGS Stats: %d persistent nodes. Root visits: %d/%d. Reuse Ratio: %.2f\n", len(m.tt), root.N, m.iterations, float64(root.N)/float64(len(m.tt)))
    
    myID := players[turnIdx]
    fmt.Printf("Estimated Winrate: %.2f%%\n", root.Q[myID]*100)
    
    bestVisits := -1
	var bestMove Move
    
    // Sort moves by visits for cleaner output
    type MoveStat struct {
        mv Move
        visits int
        winrate float64
    }
    stats := []MoveStat{}
    
    if root.Edges != nil {
        for mv, edge := range root.Edges {
            stats = append(stats, MoveStat{mv, edge.Visits, edge.Dest.Q[myID]})
            if edge.Visits > bestVisits {
                bestVisits = edge.Visits
                bestMove = mv
            }
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
    if len(stats) < limit { limit = len(stats) }
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


type BoardHash struct {
    P1, P2, P3     Bitboard
    PlayerToMoveID int
    ActiveMask     uint8
}

type PathStep struct {
    Node *MCGSNode
    Move Move
    Edge *MCGSEdge
}

func (m *MCTSPlayer) Select(root *MCGSNode) []PathStep {
    current := root
    path := []PathStep{{Node: root}}
    
    for {
        if current.untriedMoves != 0 {
            return path // Expand here
        }
        if len(current.Edges) == 0 {
            return path // Terminal
        }
        
        bestScore := math.Inf(-1)
        var bestEdge *MCGSEdge
        var bestMove Move
        
        logN := math.Log(float64(current.N + 1)) 
        
        for mv, edge := range current.Edges {
            // Q value from perspective of current player
            q := edge.Dest.Q[current.playerToMoveID]
            
            u := 2.0 * math.Sqrt(logN / float64(edge.Visits + 1))
            score := q + u
            
            if score > bestScore {
                bestScore = score
                bestEdge = edge
                bestMove = mv
            }
        }
        
        if bestEdge == nil { break } 
        
        path = append(path, PathStep{Node: bestEdge.Dest, Move: bestMove, Edge: bestEdge})
        current = bestEdge.Dest
    }
    return path
}

func (m *MCTSPlayer) Backprop(path []PathStep) {
    for i := 1; i < len(path); i++ {
        path[i].Edge.Visits++
    }
    
    for i := len(path) - 1; i >= 0; i-- {
        node := path[i].Node
        
        sumVisits := 1
        var newQ [3]float64
        
        for k := 0; k < 3; k++ {
            newQ[k] = node.U[k]
        }
        
        for _, edge := range node.Edges {
            sumVisits += edge.Visits
            for k := 0; k < 3; k++ {
                newQ[k] += edge.Dest.Q[k] * float64(edge.Visits)
            }
        }
        
        node.N = sumVisits
        for k := 0; k < 3; k++ {
            newQ[k] /= float64(node.N)
        }
        node.Q = newQ
    }
}

type MCGSNode struct {
	board            Board
	N int 
    U [3]float64 
    Q [3]float64 

    Edges map[Move]*MCGSEdge
    
	playerToMoveID   int
	remainingPlayers []int
	winnerID         int
	untriedMoves     Bitboard
}

type MCGSEdge struct {
    Dest *MCGSNode
    Visits int
}

func NewMCGSNode(board Board, playerToMoveID int, remainingPlayers []int) *MCGSNode {
	node := &MCGSNode{
		board:            board,
		playerToMoveID:   playerToMoveID,
		remainingPlayers: remainingPlayers,
		winnerID:         -1,
	}
	node.untriedMoves = node.GetPossibleMoves()
	return node
}

func (n *MCGSNode) GetPossibleMoves() Bitboard {
	if n.winnerID != -1 {
		return 0
	}

	if len(n.remainingPlayers) == 0 {
		return 0
	}
    
    found := false
    for _, id := range n.remainingPlayers {
        if id == n.playerToMoveID { found = true; break }
    }
    if !found { return 0 }

    currIdx := 0
    for i, id := range n.remainingPlayers {
        if id == n.playerToMoveID { currIdx = i; break }
    }
    nextID := n.remainingPlayers[(currIdx+1)%len(n.remainingPlayers)]

	forced := GetValidMoves(n.board, n.playerToMoveID, nextID)
	empty := ^n.board.Occupied
    myBB := n.board.GetPlayerBoard(n.playerToMoveID)
    
    wins := GetMovesThatComplete(myBB, empty, 4)
    loses := GetMovesThatComplete(myBB, empty, 3)

    if forced != 0 {
        return forced
    }
    
    targets := (empty & ^loses) | wins
    if targets == 0 {
        targets = empty & loses & ^wins
    }
    if targets == 0 {
        targets = empty
    }
    
    return targets
}

type State struct {
	board            Board
	nextPlayerID     int
	remainingPlayers []int
	winnerID         int
}

// --- Simulation Logic ---

func SimulateStep(board Board, players []int, currentID int, move Move) State {
	newBoard := board
	newBoard.Set(move.ToIndex(), currentID)

	newPlayers := make([]int, len(players))
	copy(newPlayers, players)

	pIdx := 0
	for i, id := range newPlayers {
		if id == currentID {
			pIdx = i
			break
		}
	}
    
    if CheckWin(newBoard.GetPlayerBoard(currentID)) {
        return State{board: newBoard, nextPlayerID: -1, remainingPlayers: newPlayers, winnerID: currentID}
    }
    
    nextID := -1
    
    if CheckLose(newBoard.GetPlayerBoard(currentID)) {
        newPlayers = append(newPlayers[:pIdx], newPlayers[pIdx+1:]...)
        if len(newPlayers) == 1 {
            return State{board: newBoard, nextPlayerID: -1, remainingPlayers: newPlayers, winnerID: newPlayers[0]}
        }
        if pIdx >= len(newPlayers) {
            pIdx = 0
        }
        nextID = newPlayers[pIdx]
    } else {
        nextID = newPlayers[(pIdx+1)%len(newPlayers)]
    }
    
    return State{board: newBoard, nextPlayerID: nextID, remainingPlayers: newPlayers, winnerID: -1}
}

func RunSimulation(board Board, players []int, currentID int) [3]float64 {
    simBoard := board
    simPlayers := make([]int, len(players))
    copy(simPlayers, players)
    
    curr := currentID
    
    for {
        if len(simPlayers) == 1 {
            var res [3]float64
            res[simPlayers[0]] = 1.0
            return res
        }
        
        pIdx := 0
        for i, id := range simPlayers {
            if id == curr { pIdx = i; break }
        }
        nextP := simPlayers[(pIdx+1)%len(simPlayers)]
        
        threats := GetWinningMoves(simBoard, nextP)
        myWins := GetWinningMoves(simBoard, curr)
        
        var movesBitboard Bitboard
        myBB := simBoard.GetPlayerBoard(curr)
        empty := ^simBoard.Occupied
        wins := GetMovesThatComplete(myBB, empty, 4)
        loses := GetMovesThatComplete(myBB, empty, 3)

        if threats != 0 || myWins != 0 {
            candidates := threats | myWins
            safeCandidates := (candidates & ^loses) | (candidates & wins)
            
            if safeCandidates != 0 {
                movesBitboard = safeCandidates
            } else {
                movesBitboard = candidates
            }
        } else {
            safeMoves := (empty & ^loses) | wins
            if safeMoves != 0 {
                movesBitboard = safeMoves
            } else {
                movesBitboard = empty & loses & ^wins
            }
        }
        
        if movesBitboard == 0 {
            return [3]float64{} // Draw
        }
        
        count := bits.OnesCount64(uint64(movesBitboard))
        if count == 0 { return [3]float64{} }
        
        pick := rand.Intn(count)
        var selectedIdx int
        
        temp := uint64(movesBitboard)
        for i := 0; i < pick; i++ {
              temp &= temp - 1 
        }
        selectedIdx = bits.TrailingZeros64(temp)
        
        simBoard.Set(selectedIdx, curr)
        
        if CheckWin(simBoard.GetPlayerBoard(curr)) {
            var res [3]float64
            res[curr] = 1.0
            return res
        }
        
        if CheckLose(simBoard.GetPlayerBoard(curr)) {
             newLen := len(simPlayers) - 1
             copy(simPlayers[pIdx:], simPlayers[pIdx+1:])
             simPlayers = simPlayers[:newLen]
             
             if len(simPlayers) == 1 {
                 var res [3]float64
                 res[simPlayers[0]] = 1.0
                 return res
             }
             if pIdx >= len(simPlayers) {
                 pIdx = 0
             }
             curr = simPlayers[pIdx]
        } else {
             curr = simPlayers[(pIdx+1)%len(simPlayers)]
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