import re
import statistics
from collections import Counter, defaultdict

def analyze_squava_log(filepath):
    print(f"Analyzing {filepath}...")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return

    # Split into games based on the start banner
    raw_games = content.split("Starting 3-Player Squava!")
    # Filter out empty strings (e.g. file start)
    games = [g for g in raw_games if g.strip()]

    print(f"Found {len(games)} games in log.")

    if not games:
        return

    # --- Aggregators ---
    winners = []
    win_types = []
    eliminations = {1: 0, 2: 0, 3: 0}
    game_lengths = []
    
    # Performance Stats
    dag_sizes = []
    reuse_ratios = []
    
    # Confidence (Estimated Winrate) per player
    est_winrates = defaultdict(list) 
    
    # Blunder Tracking
    # List of dicts: {game_id, move_num, player, old_wr, new_wr, diff, context}
    blunders = []
    DROP_THRESHOLD = 50.0

    for game_idx, game_data in enumerate(games):
        moves_in_game = 0
        current_player = None
        game_winner = None
        game_win_type = None
        
        # Per-game tracking
        player_last_wr = {} # {player_id: winrate_float}
        move_history = []   # list of "P1: A1" strings
        
        lines = game_data.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 1. Detect Turn
            turn_match = re.search(r"Turn: Player (\d)", line)
            if turn_match:
                current_player = int(turn_match.group(1))
                
            # 2. Detect MCGS Stats
            stats_match = re.search(r"Total Nodes in DAG: (\d+). Reuse Ratio: ([\d\.]+)", line)
            if stats_match:
                dag_sizes.append(int(stats_match.group(1)))
                reuse_ratios.append(float(stats_match.group(2)))
                
            # 3. Detect Estimated Winrate
            wr_match = re.search(r"Estimated Winrate: ([\d\.]+)%", line)
            if wr_match and current_player:
                wr = float(wr_match.group(1))
                est_winrates[current_player].append(wr)
                
                # Check for drop
                if current_player in player_last_wr:
                    old_wr = player_last_wr[current_player]
                    diff = wr - old_wr
                    if abs(diff) > DROP_THRESHOLD:
                        # Found a big drop
                        context = move_history[-3:] if len(move_history) >= 3 else move_history
                        blunders.append({
                            'game': game_idx + 1,
                            'move_idx': moves_in_game,
                            'player': current_player,
                            'old': old_wr,
                            'new': wr,
                            'diff': diff,
                            'reason': "Shift",
                            'context': context
                        })
                
                player_last_wr[current_player] = wr

            # 4. Detect Move Choice
            # "Player 1 chooses A1"
            move_match = re.search(r"Player (\d) chooses ([A-H][1-8])", line)
            if move_match:
                p_id = int(move_match.group(1))
                mv = move_match.group(2)
                move_history.append(f"P{p_id}:{mv}")
                moves_in_game += 1
                
            # 5. Detect Elimination (Loss)
            elim_match = re.search(r"Oops! Player (\d) made 3 in a row", line)
            if elim_match:
                eliminated_player = int(elim_match.group(1))
                eliminations[eliminated_player] += 1
                
                # Record elimination as drop to 0
                if eliminated_player in player_last_wr:
                    old_wr = player_last_wr[eliminated_player]
                    # Only record if it wasn't already low? No, elimination is always bad.
                    # Or if drop is significant.
                    diff = 0.0 - old_wr
                    if diff < -DROP_THRESHOLD:
                        context = move_history[-3:]
                        blunders.append({
                            'game': game_idx + 1,
                            'move_idx': moves_in_game,
                            'player': eliminated_player,
                            'old': old_wr,
                            'new': 0.0,
                            'diff': diff,
                            'reason': "Elimination",
                            'context': context
                        })
                
            # 6. Detect Win (4-in-a-row)
            win4_match = re.search(r"!!! Player (\d) wins with 4 in a row", line)
            if win4_match:
                game_winner = int(win4_match.group(1))
                game_win_type = "4-in-a-row"
                
            # 7. Detect Win (Last Standing)
            win_last_match = re.search(r"Player (\d) wins as the last player standing", line)
            if win_last_match:
                game_winner = int(win_last_match.group(1))
                game_win_type = "Last Standing"
                
            # 8. Detect Draw
            if "Game is a Draw" in line:
                game_winner = "Draw"
                game_win_type = "Draw"

        # End of game processing
        if game_winner is not None:
            winners.append(game_winner)
            win_types.append(game_win_type)
            game_lengths.append(moves_in_game)

    # --- Reporting ---
    print("\n" + "="*40)
    print(f"ANALYSIS REPORT: {len(winners)} Games Completed")
    print("="*40)
    
    # ... (Win stats etc) ...
    # 1. Win Statistics
    print("\n🏆 Win Statistics:")
    win_counts = Counter(winners)
    # Sort for consistent output: Player 1, Player 2, Player 3, Draw
    sorted_winners = sorted(win_counts.keys(), key=lambda x: str(x))
    
    for w in sorted_winners:
        count = win_counts[w]
        percentage = (count / len(winners)) * 100
        label = f"Player {w}" if isinstance(w, int) else w
        print(f"  {label:<10}: {count} wins ({percentage:.1f}%)")
        
    # 2. Win Methods
    print("\n🛑 Win Methods:")
    type_counts = Counter(win_types)
    for t, c in type_counts.items():
        print(f"  {t:<15}: {c} ({c/len(winners)*100:.1f}%)")
        
    # 3. Eliminations
    print("\n💀 Eliminations (Self-Loss via 3-in-a-row):")
    for p in [1, 2, 3]:
        print(f"  Player {p}: {eliminations[p]} times")
        
    # 4. Game Lengths
    print("\n⏱️  Game Lengths (Moves):")
    if game_lengths:
        print(f"  Average: {statistics.mean(game_lengths):.1f}")
        print(f"  Median:  {statistics.median(game_lengths):.1f}")
        print(f"  Min:     {min(game_lengths)}")
        print(f"  Max:     {max(game_lengths)}")
        
    # 5. MCGS Efficiency
    print("\n🧠 MCGS Efficiency:")
    if dag_sizes:
        print(f"  Avg DAG Size:    {statistics.mean(dag_sizes):.0f} nodes")
        print(f"  Avg Reuse Ratio: {statistics.mean(reuse_ratios):.2f}")
        print(f"  Max Reuse Ratio: {max(reuse_ratios):.2f}")
        
    # 6. Blunders
    print("\n📉 Significant Winrate Shifts (Possible Blunders > 15%):")
    if not blunders:
        print("  None detected.")
    else:
        for b in blunders:
            print(f"  Game {b['game']} Move {b['move_idx']} (P{b['player']}): {b['old']:.1f}% -> {b['new']:.1f}% ({b['diff']:.1f}%) [{b['reason']}]")
            print(f"    Context: {', '.join(b['context'])}")

    # 7. Confidence
    print("\n🤖 AI Confidence (Average Estimated Winrate):")
    for p in [1, 2, 3]:
        if est_winrates[p]:
            avg_wr = statistics.mean(est_winrates[p])
            print(f"  Player {p}: {avg_wr:.1f}%")

if __name__ == "__main__":
    import sys
    filename = "log_100k"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    analyze_squava_log(filename)
