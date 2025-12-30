package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	BoardSize  = 8
	WinLength  = 4
	LoseLength = 3
)

type Player interface {
	GetMove(board [BoardSize][BoardSize]*PlayerInfo, forcedMoves []Move) Move
	Name() string
	Symbol() string
}

type PlayerInfo struct {
	name   string
	symbol string
}

func (p *PlayerInfo) Name() string   { return p.name }
func (p *PlayerInfo) Symbol() string { return p.symbol }

type Move struct {
	r, c int
}

// --- Human Player ---

type HumanPlayer struct {
	*PlayerInfo
}

func NewHumanPlayer(name, symbol string) *HumanPlayer {
	return &HumanPlayer{PlayerInfo: &PlayerInfo{name: name, symbol: symbol}}
}

func (h *HumanPlayer) GetMove(board [BoardSize][BoardSize]*PlayerInfo, forcedMoves []Move) Move {
	reader := bufio.NewReader(os.Stdin)
	for {
		prompt := fmt.Sprintf("%s (%s), enter your move (e.g., A1): ", h.name, h.symbol)
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

		if board[r][c] != nil {
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
	*PlayerInfo
	iterations int
	game       *SquavaGame // Reference to access game state for cloning
}

func NewMCTSPlayer(name, symbol string, iterations int) *MCTSPlayer {
	return &MCTSPlayer{PlayerInfo: &PlayerInfo{name: name, symbol: symbol}, iterations: iterations}
}

func (m *MCTSPlayer) GetMove(board [BoardSize][BoardSize]*PlayerInfo, forcedMoves []Move) Move {
	// Not used directly, we use GetMoveWithContext called from Game loop
	return Move{0, 0}
}

func (m *MCTSPlayer) GetMoveWithContext(board [BoardSize][BoardSize]*PlayerInfo, forcedMoves []Move, players []*PlayerInfo, turnIdx int) Move {
	// Deep copy initial state
	initialBoard := board // Array copy is value copy? Yes, in Go arrays are values. But pointers inside are refs.
	// Wait, board contains *PlayerInfo. The pointers are shared. That's fine as PlayerInfo is immutable-ish (name/symbol const).
	// We just modify the array slots.
	
	root := NewMCTSNode(initialBoard, nil, players[turnIdx], players)

	if len(forcedMoves) > 0 {
		root.untriedMoves = forcedMoves
	}

	for i := 0; i < m.iterations; i++ {
		node := root

		// Selection
		for len(node.untriedMoves) == 0 && len(node.children) > 0 {
			node = node.UCTSelectChild()
		}

		// Expansion
		if len(node.untriedMoves) > 0 {
			move := node.untriedMoves[rand.Intn(len(node.untriedMoves))]
			state := SimulateStep(node.board, node.remainingPlayers, node.playerToMove, move)
			node = node.AddChild(move, state)
		}

		// Simulation
		var result map[string]float64
		if node.winner != nil {
			result = map[string]float64{node.winner.Symbol(): 1.0}
		} else {
			result = RunSimulation(node.board, node.remainingPlayers, node.playerToMove)
		}

		// Backprop
		for node != nil {
			node.Update(result)
			node = node.parent
		}
	}

	if len(root.children) == 0 {
		// Should not happen unless no moves
		// Fallback to random or forced
        if len(forcedMoves) > 0 {
            return forcedMoves[0]
        }
        // Find first empty
        for r := 0; r < BoardSize; r++ {
            for c := 0; c < BoardSize; c++ {
                if initialBoard[r][c] == nil {
                    return Move{r,c}
                }
            }
        }
		return Move{0, 0}
	}

	// Select best move (most visited)
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
	board            [BoardSize][BoardSize]*PlayerInfo
	parent           *MCTSNode
	children         map[Move]*MCTSNode
	visits           int
	wins             float64
	playerToMove     *PlayerInfo
	remainingPlayers []*PlayerInfo
	winner           *PlayerInfo
	untriedMoves     []Move
}

func NewMCTSNode(board [BoardSize][BoardSize]*PlayerInfo, parent *MCTSNode, playerToMove *PlayerInfo, remainingPlayers []*PlayerInfo) *MCTSNode {
	node := &MCTSNode{
		board:            board,
		parent:           parent,
		children:         make(map[Move]*MCTSNode),
		playerToMove:     playerToMove,
		remainingPlayers: remainingPlayers,
	}
	node.untriedMoves = node.GetPossibleMoves()
	return node
}

func (n *MCTSNode) GetPossibleMoves() []Move {
	if n.winner != nil || n.playerToMove == nil {
		return []Move{}
	}

	if len(n.remainingPlayers) == 0 {
		return []Move{}
	}

	currIdx := -1
	for i, p := range n.remainingPlayers {
		if p.Symbol() == n.playerToMove.Symbol() {
			currIdx = i
			break
		}
	}
	if currIdx == -1 {
		return []Move{}
	}

	nextIdx := (currIdx + 1) % len(n.remainingPlayers)
	nextPlayer := n.remainingPlayers[nextIdx]

	validMoves := StaticGetValidMoves(n.board, n.playerToMove, nextPlayer)
	if validMoves == nil {
		// All empty cells
		moves := []Move{}
		for r := 0; r < BoardSize; r++ {
			for c := 0; c < BoardSize; c++ {
				if n.board[r][c] == nil {
					moves = append(moves, Move{r, c})
				}
			}
		}
		return moves
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
	board            [BoardSize][BoardSize]*PlayerInfo
	nextPlayer       *PlayerInfo
	remainingPlayers []*PlayerInfo
	winner           *PlayerInfo
}

func (n *MCTSNode) AddChild(move Move, state State) *MCTSNode {
	child := NewMCTSNode(state.board, n, state.nextPlayer, state.remainingPlayers)
	child.winner = state.winner
	n.children[move] = child
	return child
}

func (n *MCTSNode) Update(result map[string]float64) {
	n.visits++
	if n.parent != nil {
		mover := n.parent.playerToMove
		if val, ok := result[mover.Symbol()]; ok {
			n.wins += val
		}
	}
}

// --- Simulation Logic ---

func SimulateStep(board [BoardSize][BoardSize]*PlayerInfo, players []*PlayerInfo, currentPlayer *PlayerInfo, move Move) State {
	newBoard := board // Value copy of array
	newBoard[move.r][move.c] = currentPlayer

	newPlayers := make([]*PlayerInfo, len(players))
	copy(newPlayers, players)

	pIdx := 0
	for i, p := range newPlayers {
		if p.Symbol() == currentPlayer.Symbol() {
			pIdx = i
			break
		}
	}

	if StaticCheckWinCondition(newBoard, move.r, move.c, currentPlayer) {
		return State{board: newBoard, nextPlayer: nil, remainingPlayers: newPlayers, winner: currentPlayer}
	}

	var nextPlayer *PlayerInfo

	if StaticCheckLoseCondition(newBoard, move.r, move.c, currentPlayer) {
		// Eliminate
		newPlayers = append(newPlayers[:pIdx], newPlayers[pIdx+1:]...)
		if len(newPlayers) == 1 {
			return State{board: newBoard, nextPlayer: nil, remainingPlayers: newPlayers, winner: newPlayers[0]}
		}
		if pIdx >= len(newPlayers) {
			pIdx = 0
		}
		nextPlayer = newPlayers[pIdx]
	} else {
		nextIdx := (pIdx + 1) % len(newPlayers)
		nextPlayer = newPlayers[nextIdx]
	}

	return State{board: newBoard, nextPlayer: nextPlayer, remainingPlayers: newPlayers, winner: nil}
}

func RunSimulation(board [BoardSize][BoardSize]*PlayerInfo, players []*PlayerInfo, currentPlayer *PlayerInfo) map[string]float64 {
	simBoard := board
	simPlayers := make([]*PlayerInfo, len(players))
	copy(simPlayers, players)

	if len(simPlayers) == 1 {
		return map[string]float64{simPlayers[0].Symbol(): 1.0}
	}

	curr := currentPlayer

	for {
		if len(simPlayers) == 1 {
			return map[string]float64{simPlayers[0].Symbol(): 1.0}
		}

		pIdx := -1
		for i, p := range simPlayers {
			if p.Symbol() == curr.Symbol() {
				pIdx = i
				break
			}
		}
		nextP := simPlayers[(pIdx+1)%len(simPlayers)]

		validMoves := StaticGetValidMoves(simBoard, curr, nextP)
		var moves []Move
		if validMoves == nil {
			for r := 0; r < BoardSize; r++ {
				for c := 0; c < BoardSize; c++ {
					if simBoard[r][c] == nil {
						moves = append(moves, Move{r, c})
					}
				}
			}
		} else {
			moves = validMoves
		}

		if len(moves) == 0 {
			return map[string]float64{} // Draw
		}

		move := moves[rand.Intn(len(moves))]
		state := SimulateStep(simBoard, simPlayers, curr, move)

		simBoard = state.board
		simPlayers = state.remainingPlayers

		if state.winner != nil {
			return map[string]float64{state.winner.Symbol(): 1.0}
		}

		if len(simPlayers) == 0 {
			return map[string]float64{}
		}
		
		curr = state.nextPlayer
        if curr == nil {
             // Should have been winner or handled above
             break
        }
	}
	return map[string]float64{}
}

// --- Game Logic Static Methods ---

func CountConsecutive(board [BoardSize][BoardSize]*PlayerInfo, r, c, dr, dc int, symbol string) int {
	count := 1
	nr, nc := r+dr, c+dc
	for isValidCoord(nr, nc) && board[nr][nc] != nil && board[nr][nc].Symbol() == symbol {
		count++
		nr += dr
		nc += dc
	}
	nr, nc = r-dr, c-dc
	for isValidCoord(nr, nc) && board[nr][nc] != nil && board[nr][nc].Symbol() == symbol {
		count++
		nr -= dr
		nc -= dc
	}
	return count
}

func StaticCheckWinCondition(board [BoardSize][BoardSize]*PlayerInfo, r, c int, player *PlayerInfo) bool {
	directions := [][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}
	for _, d := range directions {
		if CountConsecutive(board, r, c, d[0], d[1], player.Symbol()) >= WinLength {
			return true
		}
	}
	return false
}

func StaticCheckLoseCondition(board [BoardSize][BoardSize]*PlayerInfo, r, c int, player *PlayerInfo) bool {
	directions := [][2]int{{0, 1}, {1, 0}, {1, 1}, {1, -1}}
	for _, d := range directions {
		if CountConsecutive(board, r, c, d[0], d[1], player.Symbol()) >= LoseLength {
			return true
		}
	}
	return false
}

func StaticGetWinningMoves(board [BoardSize][BoardSize]*PlayerInfo, player *PlayerInfo) []Move {
	moves := []Move{}
	for r := 0; r < BoardSize; r++ {
		for c := 0; c < BoardSize; c++ {
			if board[r][c] == nil {
				board[r][c] = player
				if StaticCheckWinCondition(board, r, c, player) {
					moves = append(moves, Move{r, c})
				}
				board[r][c] = nil
			}
		}
	}
	return moves
}

func StaticGetValidMoves(board [BoardSize][BoardSize]*PlayerInfo, currentPlayer, nextPlayer *PlayerInfo) []Move {
	threats := StaticGetWinningMoves(board, nextPlayer)
	myWins := StaticGetWinningMoves(board, currentPlayer)

	if len(threats) == 0 {
		return nil
	}

	validSet := make(map[Move]bool)
	for _, m := range threats {
		validSet[m] = true
	}
	for _, m := range myWins {
		validSet[m] = true
	}

	validList := []Move{}
	for m := range validSet {
		validList = append(validList, m)
	}
	return validList
}

// --- Squava Game Engine ---

type SquavaGame struct {
	board   [BoardSize][BoardSize]*PlayerInfo
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
			if g.board[r][c] != nil {
				symbol = g.board[r][c].Symbol()
			}
			fmt.Printf("%s ", symbol)
		}
		fmt.Println()
	}
}

func (g *SquavaGame) IsBoardFull() bool {
	for r := 0; r < BoardSize; r++ {
		for c := 0; c < BoardSize; c++ {
			if g.board[r][c] == nil {
				return false
			}
		}
	}
	return true
}

func (g *SquavaGame) Run() {
	fmt.Println("Starting 3-Player Squava!")
	fmt.Println("Board Size: 8x8")
	fmt.Println("Rules: 4-in-a-row wins. 3-in-a-row loses.")

	// For MCTS Context need proper PlayerInfo pointers that match board
	// The Players wrap PlayerInfo.
	playerInfos := []*PlayerInfo{}
	for _, p := range g.players {
		// Reflection or interface method to get info?
		// We added Name/Symbol accessor.
		// We need to cast to concrete to get struct pointer if we want equality checks to work by pointer?
		// Actually equality check by Symbol string is safer.
		playerInfos = append(playerInfos, &PlayerInfo{name: p.Name(), symbol: p.Symbol()})
	}

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

		// To use static methods, we need PlayerInfos matching these players
		pInfo := playerInfos[0] // find matching
		nextPInfo := playerInfos[0]
		for _, pi := range playerInfos {
			if pi.Symbol() == currentPlayer.Symbol() {
				pInfo = pi
			}
			if pi.Symbol() == nextPlayer.Symbol() {
				nextPInfo = pi
			}
		}

		fmt.Printf("Turn: %s (%s)\n", currentPlayer.Name(), currentPlayer.Symbol())

		forcedMoves := StaticGetValidMoves(g.board, pInfo, nextPInfo)

		var move Move
		if mcts, ok := currentPlayer.(*MCTSPlayer); ok {
			fmt.Printf("%s is thinking...\n", currentPlayer.Name())
			// Filter playerInfos to currently active players
			activeInfos := []*PlayerInfo{}
			for _, ap := range g.players {
				for _, pi := range playerInfos {
					if pi.Symbol() == ap.Symbol() {
						activeInfos = append(activeInfos, pi)
					}
				}
			}
			
			// Re-calculate turnIdx relative to active players
			activeTurnIdx := 0
			for i, ap := range activeInfos {
				if ap.Symbol() == currentPlayer.Symbol() {
					activeTurnIdx = i
					break
				}
			}
			
			move = mcts.GetMoveWithContext(g.board, forcedMoves, activeInfos, activeTurnIdx)
			fmt.Printf("%s chooses %c%d\n", currentPlayer.Name(), move.c+65, move.r+1)
		} else {
			g.PrintBoard()
			move = currentPlayer.GetMove(g.board, forcedMoves)
		}

		g.board[move.r][move.c] = pInfo

		if StaticCheckWinCondition(g.board, move.r, move.c, pInfo) {
			g.PrintBoard()
			fmt.Printf("!!! %s wins with 4 in a row! !!!\n", currentPlayer.Name())
			return
		}

		if StaticCheckLoseCondition(g.board, move.r, move.c, pInfo) {
			fmt.Printf("Oops! %s made 3 in a row and is eliminated!\n", currentPlayer.Name())
			// Remove player
			g.players = append(g.players[:g.turnIdx], g.players[g.turnIdx+1:]...)
			if g.turnIdx >= len(g.players) {
				g.turnIdx = 0
			}
			
			if g.IsBoardFull() {
				g.PrintBoard()
				fmt.Println("Board full! Game is a Draw between remaining players.")
				return
			}
			continue
		}

		if g.IsBoardFull() {
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
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	game := NewSquavaGame()

	createPlayer := func(t, name, symbol string) Player {
		if t == "mcts" {
			return NewMCTSPlayer(name, symbol, *iterations)
		}
		return NewHumanPlayer(name, symbol)
	}

	game.AddPlayer(createPlayer(*p1Type, "Player 1", "X"))
	game.AddPlayer(createPlayer(*p2Type, "Player 2", "O"))
	game.AddPlayer(createPlayer(*p3Type, "Player 3", "Z"))

	game.Run()
}
