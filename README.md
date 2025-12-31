# Squava for 3 players

Proposed by [Craig Duncan on BGG](https://boardgamegeek.com/thread/1315703/three-player-squava)

A high-performance, 3-player implementation of Squava on an 8x8 board, written in Go by Gemini.

## The Game

Squava is traditionally a 2-player game. This version expands the rules for 3 players:
- **Winning:** The first player to get **4-in-a-row** (horizontally, vertically, or diagonally) wins immediately.
- **Elimination:** If a player creates a **3-in-a-row** and does *not* complete a 4-in-a-row with the same move, they are eliminated from the game. Their pieces remain on the board as obstacles.
- **Goal:** Be the last player standing or the first to complete 4-in-a-row.

## Technical Architecture

### Bitboard Engine
The game state is represented using three 64-bit integers (`uint64`), one for each player. This allows for extremely fast move generation and win/loss detection using constant-time bitwise operations.

- **Unified Bitwise Generation:** Uses parallel shifts and masks to detect 3-in-a-row and 4-in-a-row patterns across all four directions (Horizontal, Vertical, and both Diagonals) without iterating over the board.
- **Zero-Allocation Hot Path:** The core search and simulation logic uses fixed-size arrays (`[3]float64`) instead of maps to track player scores, eliminating garbage collection pressure during high-iteration MCTS runs.

### AI: Monte Carlo Graph Search (MCGS)
The AI utilizes Monte Carlo Tree Search expanded into a Directed Acyclic Graph (DAG) via a Transposition Table.

- **Persistent DAG:** Each AI player maintains its search graph throughout the game. Turn-to-turn results are preserved, allowing the AI to "think" deeper as the game progresses by reusing previously explored paths.
- **Target-Based Iteration:** The search continues until the root node (the current board state) reaches a specific visit threshold (default: 1,000 iterations), ensuring consistent depth regardless of how many nodes were reused.
- **Transposition Table:** Game states are hashed using player bitboards and an active player bitmask, allowing the AI to recognize identical states reached through different move orders.

## Performance Tuning

The engine is optimized for high throughput:
- **Loop Unrolling:** Critical move generation paths are unrolled to maximize instruction-level parallelism.
- **Suicide Pruning:** The simulation (rollout) phase proactively avoids moves that lead to immediate elimination (3-in-a-row) unless no other moves are possible.
- **Forced Move Detection:** Automatically identifies moves required to block an opponent's immediate win.

## Usage

### Build
```bash
go build squava.go
```

### Run
```bash
# 3 Humans
./squava

# 1 Human vs 2 AI (10,000 iterations)
./squava -p2 mcts -p3 mcts -iterations 10000

# AI vs AI vs AI for profiling
./squava -p1 mcts -p2 mcts -p3 mcts -iterations 100000 -cpuprofile cpu.prof
```

### Flags
- `-p1, -p2, -p3`: Player type (`human` or `mcts`).
- `-iterations`: Number of visits the root node must reach per turn.
- `-cpuprofile`: File path to write a CPU profile for performance analysis.

### Profiling and Analysis

To analyze the performance of the engine, you can generate a CPU profile and view it using Go's built-in profiling tools:

1. **Generate a profile:**
   ```bash
   ./squava -p1 mcts -p2 mcts -p3 mcts -iterations 100000 -cpuprofile cpu.prof
   ```

2. **View the top bottlenecks in the CLI:**
   ```bash
   go tool pprof -top cpu.prof
   ```

3. **Open an interactive web interface (requires Graphviz):**
   ```bash
   go tool pprof -http=:8080 cpu.prof
   ```
   This will open a browser window showing a call graph and a flame graph of the execution.

## Performance Benchmarks

Based on an analysis of 100 games (1,000,000 iterations per turn):

### Win Statistics
- **Player 1 (X):** 32.0% wins
- **Player 2 (O):** 36.0% wins
- **Player 3 (Z):** 31.0% wins
- **Draw:** 1.0%

### Game Dynamics
- **Average Game Length:** 40.7 moves
- **Primary Win Method:** Last Standing (73.0%)
- **Eliminations:** All players show similar elimination rates (P1: 60, P2: 53, P3: 55), suggesting a balanced risk profile at higher iteration counts.

### AI Behavior
- **Average Confidence:** AI players maintain a balanced average estimated winrate (~34%).
- **Blunder Detection:** The engine detects significant winrate shifts (blunders > 50%), which now occur more frequently in complex mid-game transitions rather than early-game forced moves.

