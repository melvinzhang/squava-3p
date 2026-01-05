# Squava for 3 players

Proposed by [Craig Duncan on BGG](https://boardgamegeek.com/thread/1315703/three-player-squava)

A high-performance, 3-player implementation of Squava on an 8x8 board, written in Go by Gemini.

## The Game

Squava is traditionally a 2-player game. This version expands the rules for 3 players:
- **Winning:** The first player to get **4-in-a-row** (horizontally, vertically, or diagonally) wins immediately.
- **Elimination:** If a player creates a **3-in-a-row** and does *not* complete a 4-in-a-row with the same move, they are eliminated from the game. Their pieces remain on the board as obstacles.
- **Forced Moves:** If a player can complete a 4-in-a-row on their current turn, they **must** take the win. If they cannot win immediately but the **next** active player has a winning move, the current player is **forced** to block that move.
- **Goal:** Be the last player standing or the first to complete 4-in-a-row.

## Web Version

The game is also available as a fully client-side web application. It uses the same high-performance Go engine compiled to WebAssembly.

### Architecture
- **WebAssembly (WASM):** The Go engine is compiled to WASM using the `js/wasm` target. It utilizes a pure Go fallback for bitwise operations since AVX2 is not available in the browser.
- **Web Workers:** To prevent UI freezing during deep MCTS searches (20,000+ iterations), the WASM engine runs inside a dedicated Web Worker.
- **Automated AI:** You can choose to play as any of the three players. The engine automatically triggers AI moves for the other two participants.

### Running the Web Version
1. **Build and Serve:**
   ```bash
   make serve
   ```
2. **Access:** Open `http://localhost:8080` in your browser.

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

### Commands

| Command | Description |
|---------|-------------|
| `make build` | Compiles the `squava` binary. |
| `make test` | Runs the test suite. |
| `make fuzz` | Runs fuzz tests for robustness. |
| `make benchmark` | Runs 100 MCTS games and saves logs to `logs/`. |
| `make analyze` | Analyzes benchmark logs to produce win/loss statistics. |
| `make profile` | Runs a high-iteration game and outputs a CPU profile. |
| `make clean` | Removes binaries and profile files. |

### Running the Game

After building with `make build`, you can run the game:

```bash
# 3 Humans
./squava

# 1 Human vs 2 AI (10,000 iterations)
./squava -p2 mcts -p3 mcts -iterations 10000

# AI vs AI vs AI with a specific seed
./squava -p1 mcts -p2 mcts -p3 mcts -iterations 1000000 -seed 641728870
```

### Flags
- `-p1, -p2, -p3`: Player type (`human` or `mcts`).
- `-iterations`: Number of visits the root node must reach per turn.
- `-seed`: Random seed for reproducibility.
- `-cpuprofile`: File path to write a CPU profile for performance analysis.

## Profiling and Analysis

To analyze the performance of the engine, use the built-in profiling rules:

1. **Generate and view a top-level profile:**
   ```bash
   make profile
   ```

2. **Inspect specific functions:**
   ```bash
   make pprof
   ```

3. **Open an interactive web interface (requires Graphviz):**
   ```bash
   go tool pprof -http=:8080 cpu.prof
   ```

## Performance Benchmarks

Based on an analysis of 426 games (1,000,000 iterations per turn):

### Win Statistics
- **Player 1 (X):** 10.1% wins
- **Player 2 (O):** 28.4% wins
- **Player 3 (Z):** 40.6% wins
- **Draw:** 20.9%

### Game Dynamics
- **Average Game Length:** 55.4 moves
- **Primary Win Methods:** Last Standing (75.1%), Draw (20.9%), 4-in-a-row (4.0%)
- **Eliminations:** Player 1 shows a higher elimination rate (P1: 321, P2: 231, P3: 179), likely due to the disadvantage of moving first in this 3-player configuration.

### AI Behavior
- **Average Confidence:** AI players' estimated winrates (P1: 25.6%, P2: 39.9%, P3: 39.6%) align closely with actual win outcomes.
- **Blunder Detection:** The engine detects significant winrate shifts (blunders > 51%), typically occurring during complex mid-game transitions where forced blocks and multiple threats overlap.

