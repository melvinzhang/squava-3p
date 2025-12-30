# Squava (3-Player)

A high-performance implementation of the board game **Squava** for 3 players, featuring a strong AI agent powered by **Monte Carlo Graph Search (MCGS)**.

## The Game

Squava is a strategy game usually played on a square grid. While traditionally a 2-player game (Yavalath), this implementation adapts it for 3 players on an 8x8 board.

### Rules
1.  **Objective:** The goal is to place **4** stones in a row (horizontally, vertically, or diagonally).
2.  **Losing Condition:** If you place **3** stones in a row, you lose and are immediately eliminated from the game.
3.  **Precedence:** If a move completes both a line of 4 (Win) and a line of 3 (Loss) simultaneously, the **Win** counts. You win.
4.  **Forced Blocks:** If the player acting immediately after you has a winning move available, you **must** play to block them (unless you can win immediately yourself).

## Implementation Details

This project is written in **Go** to leverage its raw performance and concurrency features for the AI search.

### Core Technologies
*   **Monte Carlo Graph Search (MCGS):** Instead of a standard Tree Search, the AI builds a **Directed Acyclic Graph (DAG)**. It uses a **Transposition Table** to detect when different sequences of moves result in the same board state. This allows different branches of the search to share statistics and value estimates, providing an exponential efficiency boost compared to standard MCTS.
*   **Bitboards:** The 8x8 board fits perfectly into 64-bit integers. The state is tracked using three `uint64` bitboards (one per player). Win/Loss checks and move generation are performed using **O(1) bitwise operations** (shifts and masks) rather than iterating over arrays.
*   **Suicide Pruning:** The move generator automatically filters out moves that result in immediate self-elimination (3-in-a-row) unless such a move is forced or leads to an immediate win.

### Performance
The combination of MCGS and Bitboards allows the AI to perform tens of thousands of simulations in seconds. As the game progresses and the board fills up, the **Reuse Ratio** (iterations / unique nodes) skyrockets, allowing the AI to "solve" complex late-game situations deeply.

## Usage

### Prerequisites
*   [Go](https://go.dev/dl/) installed on your machine.

### Running the Game
Run the source code directly:

```bash
go run squava.go [flags]
```

### Configuration Flags
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--p1`, `--p2`, `--p3` | Set player type. Options: `human`, `mcts`. | `human` |
| `--iterations` | Number of simulations per AI move. | `1000` |
| `--cpuprofile` | Write a CPU profile to a file for debugging. | *(empty)* |

### Examples

**Play as Player 1 against two AI opponents (5000 iterations):**
```bash
go run squava.go --p1 human --p2 mcts --p3 mcts --iterations 5000
```

**Watch three AIs play against each other:**
```bash
go run squava.go --p1 mcts --p2 mcts --p3 mcts --iterations 10000
```

**Debug performance:**
```bash
go run squava.go --p1 mcts --p2 mcts --p3 mcts --cpuprofile cpu.prof
go tool pprof -top cpu.prof
```

## AI Stats Key
When the AI thinks, it outputs statistics:
*   **Reuse Ratio:** How many times the simulation hit an existing node in the graph. High numbers (e.g., >100) indicate the DAG is working effectively.
*   **Estimated Winrate:** The AI's confidence in winning from the current position.
*   **Top Moves:** A breakdown of the most visited moves and their specific win rates.

## Future work
1. Persistent Transposition Table: Knowledge is now stored in MCTSPlayer, allowing the AI to "remember" and reuse analysis across turns.
2. Zero-Allocation Search: All map[int]float64 objects have been replaced with [3]float64 arrays. This significantly reduces garbage collection pauses.
3. Unified Bitwise Move Generation: A new helper GetMovesThatComplete(length int) handles both win-detection (length 4) and suicide-prevention (length 3) using
   constant-time bitwise shifts.
4. Optimized Rollout Policy: The simulation loop now uses the bitwise generators to avoid immediate suicide and respect forced blocks without iterating over the board
   bits.
