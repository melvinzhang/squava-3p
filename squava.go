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
	WinLength  = 4
	LoseLength = 3
)

// Bitboard constants
const (
	FileA uint64 = 0x0101010101010101
	FileB uint64 = FileA << 1
	FileC uint64 = FileA << 2
	FileD uint64 = FileA << 3
	FileE uint64 = FileA << 4
	FileF uint64 = FileA << 5
	FileG uint64 = FileA << 6
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
	GetMove(board Board, forcedMoves []Move) Move
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
	myBB := board.GetPlayerBoard(pID)
	empty := ^board.Occupied
	var threats Bitboard

	r1 := (myBB >> 1) & Bitboard(^FileH)
	r2 := (myBB >> 2) & Bitboard(^(FileH | FileG))
	r3 := (myBB >> 3) & Bitboard(^(FileH | FileG | FileF))
	
	t := myBB & r1 & r2
	threats |= (t << 3) & Bitboard(^(FileA | FileB | FileC))
	
	threats |= r1 & r2 & r3
	
	t = myBB & r1 & r3
	threats |= (t << 2) & Bitboard(^(FileA | FileB))
	
	t = myBB & r2 & r3
	threats |= (t << 1) & Bitboard(^FileA)
	
	r1 = myBB >> 8
	r2 = myBB >> 16
	r3 = myBB >> 24
	
t = myBB & r1 & r2
	threats |= t << 24
	
threats |= r1 & r2 & r3
	
t = myBB & r1 & r3
	threats |= t << 16
	
t = myBB & r2 & r3
	threats |= t << 8
	
r1 = (myBB >> 9) & Bitboard(^FileH)
	r2 = (myBB >> 18) & Bitboard(^(FileH | FileG))
	r3 = (myBB >> 27) & Bitboard(^(FileH | FileG | FileF))
	
t = myBB & r1 & r2
	threats |= (t << 27) & Bitboard(^(FileA | FileB | FileC))
	
threats |= r1 & r2 & r3
	
t = myBB & r1 & r3
	threats |= (t << 18) & Bitboard(^(FileA | FileB))
	
t = myBB & r2 & r3
	threats |= (t << 9) & Bitboard(^FileA)

	r1 = (myBB >> 7) & Bitboard(^FileA)
	r2 = (myBB >> 14) & Bitboard(^(FileA | FileB))
	r3 = (myBB >> 21) & Bitboard(^(FileA | FileB | FileC))
	
t = myBB & r1 & r2
	threats |= (t << 21) & Bitboard(^(FileH | FileG | FileF))
	
threats |= r1 & r2 & r3
	
t = myBB & r1 & r3
	threats |= (t << 14) & Bitboard(^(FileH | FileG))
	
t = myBB & r2 & r3
	threats |= (t << 7) & Bitboard(^FileH)
	
	return threats & empty
}

func GetValidMoves(board Board, currentPID, nextPID int) []Move {
	threats := GetWinningMoves(board, nextPID)
	myWins := GetWinningMoves(board, currentPID)
	
	if threats == 0 {
		return nil
	}
	
	combined := threats | myWins
	
	moves := []Move{}
	for combined != 0 {
		idx := bits.TrailingZeros64(uint64(combined))
		moves = append(moves, MoveFromIndex(idx))
		combined &= Bitboard(^(uint64(1) << idx)) // Corrected: Use Bitboard for the mask
	}
	return moves
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

func (h *HumanPlayer) GetMove(board Board, forcedMoves []Move) Move {
	reader := bufio.NewReader(os.Stdin)
	for {
		prompt := fmt.Sprintf("%s (%s), enter your move (e.g., A1): ", h.info.name, h.info.symbol)
		if len(forcedMoves) > 0 {
			forcedStr := []string{}
			for _, m := range forcedMoves {
				forcedStr = append(forcedStr, fmt.Sprintf("%c%d", m.c+65, m.r+1))
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
		if len(forcedMoves) > 0 && !containsMove(forcedMoves, move) {
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

func containsMove(moves []Move, m Move) bool {
	for _, v := range moves {
		if v == m {
			return true
		}
	}
	return false
}

// --- MCTS Player ---

type MCTSPlayer struct {
	info       PlayerInfo
	iterations int
}

func NewMCTSPlayer(name, symbol string, id int, iterations int) *MCTSPlayer {
	return &MCTSPlayer{info: PlayerInfo{name: name, symbol: symbol, id: id}, iterations: iterations}
}
func (m *MCTSPlayer) Name() string { return m.info.name }
func (m *MCTSPlayer) Symbol() string { return m.info.symbol }
func (m *MCTSPlayer) ID() int { return m.info.id }

func (m *MCTSPlayer) GetMove(board Board, forcedMoves []Move) Move {
    return Move{0,0}
}

func (m *MCTSPlayer) GetMoveWithContext(board Board, forcedMoves []Move, players []int, turnIdx int) Move {
	root := NewMCTSNode(board, nil, players[turnIdx], players)

	if len(forcedMoves) > 0 {
		root.untriedMoves = forcedMoves
	}

	for i := 0; i < m.iterations; i++ {
		node := root

		for len(node.untriedMoves) == 0 && len(node.children) > 0 {
			node = node.UCTSelectChild()
		}

		if len(node.untriedMoves) > 0 {
			move := node.untriedMoves[rand.Intn(len(node.untriedMoves))]
			state := SimulateStep(node.board, node.remainingPlayers, node.playerToMoveID, move)
			node = node.AddChild(move, state)
		}

		var result map[int]float64
		if node.winnerID != -1 {
			result = map[int]float64{node.winnerID: 1.0}
		} else {
			result = RunSimulation(node.board, node.remainingPlayers, node.playerToMoveID)
		}

		for node != nil {
			node.Update(result)
			node = node.parent
		}
	}

	if len(root.children) == 0 {
        if len(forcedMoves) > 0 { return forcedMoves[0] }
        empty := ^board.Occupied
        if empty != 0 {
            idx := bits.TrailingZeros64(uint64(empty))
            return MoveFromIndex(idx)
        }
		return Move{0, 0}
	}

	bestVisits := -1
	var bestMove Move
	for m, child := range root.children {
		if child.visits > bestVisits {
			bestVisits = child.visits
			bestMove = m
		}
	}
	return bestMove
}

type MCTSNode struct {
	board            Board
	parent           *MCTSNode
	children         map[Move]*MCTSNode
	visits           int
	wins             float64
	playerToMoveID   int
	remainingPlayers []int
	winnerID         int
	untriedMoves     []Move
}

func NewMCTSNode(board Board, parent *MCTSNode, playerToMoveID int, remainingPlayers []int) *MCTSNode {
	node := &MCTSNode{
		board:            board,
		parent:           parent,
		children:         make(map[Move]*MCTSNode),
		playerToMoveID:   playerToMoveID,
		remainingPlayers: remainingPlayers,
		winnerID:         -1,
	}
	node.untriedMoves = node.GetPossibleMoves()
	return node
}

func (n *MCTSNode) GetPossibleMoves() []Move {
	if n.winnerID != -1 {
		return []Move{}
	}

	if len(n.remainingPlayers) == 0 {
		return []Move{}
	}
    
    found := false
    for _, id := range n.remainingPlayers {
        if id == n.playerToMoveID { found = true; break }
    }
    if !found { return []Move{} }

    currIdx := 0
    for i, id := range n.remainingPlayers {
        if id == n.playerToMoveID { currIdx = i; break }
    }
    nextID := n.remainingPlayers[(currIdx+1)%len(n.remainingPlayers)]

	validMoves := GetValidMoves(n.board, n.playerToMoveID, nextID)
	if validMoves == nil {
		empty := ^n.board.Occupied
        myBB := n.board.GetPlayerBoard(n.playerToMoveID)
        
        safeMoves := []Move{}
        suicideMoves := []Move{}
        
        for empty != 0 {
            idx := bits.TrailingZeros64(uint64(empty))
            move := MoveFromIndex(idx)
            mask := Bitboard(1) << idx
            
            // Check if move wins or loses
            newBB := myBB | mask
            isWin := CheckWin(newBB)
            isLose := CheckLose(newBB)
            
            if isWin {
                safeMoves = append(safeMoves, move)
            } else if isLose {
                suicideMoves = append(suicideMoves, move)
            } else {
                safeMoves = append(safeMoves, move)
            }
            
            empty &= Bitboard(^(uint64(1) << idx))
        }
        
        if len(safeMoves) > 0 {
            return safeMoves
        }
		return suicideMoves
	}
	return validMoves
}

func (n *MCTSNode) UCTSelectChild() *MCTSNode {
	logVisits := math.Log(float64(n.visits))
	bestScore := math.Inf(-1)
	var bestChild *MCTSNode

	for _, child := range n.children {
		score := math.Inf(1)
		if child.visits > 0 {
			winRate := child.wins / float64(child.visits)
			explore := math.Sqrt(2 * logVisits / float64(child.visits))
			score = winRate + explore
		}

		if score > bestScore {
			bestScore = score
			bestChild = child
		}
	}
	return bestChild
}

type State struct {
	board            Board
	nextPlayerID     int
	remainingPlayers []int
	winnerID         int
}

func (n *MCTSNode) AddChild(move Move, state State) *MCTSNode {
	child := NewMCTSNode(state.board, n, state.nextPlayerID, state.remainingPlayers)
	child.winnerID = state.winnerID
	n.children[move] = child
	return child
}

func (n *MCTSNode) Update(result map[int]float64) {
	n.visits++
	if n.parent != nil {
		mover := n.parent.playerToMoveID
		if val, ok := result[mover]; ok {
			n.wins += val
		}
	}
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

func RunSimulation(board Board, players []int, currentID int) map[int]float64 {
    simBoard := board
    simPlayers := make([]int, len(players))
    copy(simPlayers, players)
    
    curr := currentID
    
    for {
        if len(simPlayers) == 1 {
            return map[int]float64{simPlayers[0]: 1.0}
        }
        
        pIdx := 0
        for i, id := range simPlayers {
            if id == curr { pIdx = i; break }
        }
        nextP := simPlayers[(pIdx+1)%len(simPlayers)]
        
        threats := GetWinningMoves(simBoard, nextP)
        myWins := GetWinningMoves(simBoard, curr)
        
        var movesBitboard Bitboard
        if threats != 0 {
            movesBitboard = threats | myWins
        } else {
            movesBitboard = ^simBoard.Occupied
        }
        
        if movesBitboard == 0 {
            return map[int]float64{} // Draw
        }
        
        count := bits.OnesCount64(uint64(movesBitboard))
        if count == 0 { return map[int]float64{} }
        
        pick := rand.Intn(count)
        var selectedIdx int
        
        temp := uint64(movesBitboard)
        for i := 0; i < pick; i++ {
              temp &= temp - 1 
        }
        selectedIdx = bits.TrailingZeros64(temp)
        
        simBoard.Set(selectedIdx, curr)
        
        if CheckWin(simBoard.GetPlayerBoard(curr)) {
            return map[int]float64{curr: 1.0}
        }
        
        if CheckLose(simBoard.GetPlayerBoard(curr)) {
             newLen := len(simPlayers) - 1
             copy(simPlayers[pIdx:], simPlayers[pIdx+1:])
             simPlayers = simPlayers[:newLen]
             
             if len(simPlayers) == 1 {
                 return map[int]float64{simPlayers[0]: 1.0}
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