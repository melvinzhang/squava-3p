import re
import statistics
from collections import Counter, defaultdict
import sys
import glob

def analyze_squava_logs(filepaths):
    print(f"Analyzing {len(filepaths)} log files...")
    
    # --- Aggregators ---
    winners = []
    win_types = []
    eliminations = {1: 0, 2: 0, 3: 0}
    game_lengths = []
    
    # Confidence (Estimated Winrate) per player
    est_winrates = defaultdict(list) 
    
    # Blunder Tracking
    # List of dicts: {seed, move_num, player, old_wr, new_wr, diff, context}
    blunders = []
    DROP_THRESHOLD = 51.0
    
    games_processed = 0

    for filepath in filepaths:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"File '{filepath}' not found.")
            continue

        # Each file is expected to contain one game, but we'll use the banner just in case
        # or treat the whole file as one game if banner is missing but content exists
        raw_games = content.split("Starting 3-Player Squava!")
        # Filter out empty strings (e.g. file start)
        games = [g for g in raw_games if g.strip()]
        
        if not games:
            # Fallback: maybe the file is just the log without the banner?
            if content.strip():
                games = [content]
            else:
                continue

        for game_data in games:
            games_processed += 1
            moves_in_game = 0
            current_move_num = 0
            current_player = None
            game_winner = None
            game_win_type = None
            seed = "Unknown"
            
            # Per-game tracking
            player_last_wr = {} # {player_id: winrate_float}
            move_history = []   # list of "P1: A1" strings
            
            lines = game_data.strip().split('\n')
            
            for line in lines:
                line = line.strip()

                # 0. Detect Seed
                seed_match = re.search(r"Random Seed: (\d+)", line)
                if seed_match:
                    seed = seed_match.group(1)
                
                # 1. Detect Turn
                turn_match = re.search(r"Move (\d+): Player (\d)", line)
                if turn_match:
                    current_move_num = int(turn_match.group(1))
                    current_player = int(turn_match.group(2))
                    
                # 2. Detect Estimated Winrate
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
                                'seed': seed,
                                'move_idx': current_move_num,
                                'player': current_player,
                                'old': old_wr,
                                'new': wr,
                                'diff': diff,
                                'reason': "Shift",
                                'context': context
                            })
                    
                    player_last_wr[current_player] = wr

                # 3. Detect Move Choice
                # "Player 1 chooses A1"
                move_match = re.search(r"Player (\d) chooses ([A-H][1-8])", line)
                if move_match:
                    p_id = int(move_match.group(1))
                    mv = move_match.group(2)
                    move_history.append(f"P{p_id}:{mv}")
                    moves_in_game += 1
                    
                # 4. Detect Result (Win/Elimination/Draw)
                result_match = re.search(r"Result: (.*)", line)
                if result_match:
                    content = result_match.group(1)
                    
                    if "Draw" in content:
                        game_winner = "Draw"
                        game_win_type = "Draw"
                        
                    elif "Wins" in content:
                        # e.g. "Player 1 Wins (4-in-a-row)"
                        p_match = re.search(r"Player (\d)", content)
                        type_match = re.search(r"\((.*)\)", content)
                        
                        if p_match:
                            game_winner = int(p_match.group(1))
                        if type_match:
                            game_win_type = type_match.group(1)
                            
                    elif "Eliminated" in content:
                        # e.g. "Player 1 Eliminated (3-in-a-row)"
                        p_match = re.search(r"Player (\d)", content)
                        if p_match:
                            eliminated_player = int(p_match.group(1))
                            eliminations[eliminated_player] += 1
                            
                            # Record elimination as drop to 0
                            if eliminated_player in player_last_wr:
                                old_wr = player_last_wr[eliminated_player]
                                diff = 0.0 - old_wr
                                if abs(diff) > DROP_THRESHOLD:
                                    context = move_history[-3:]
                                    blunders.append({
                                        'seed': seed,
                                        'move_idx': current_move_num,
                                        'player': eliminated_player,
                                        'old': old_wr,
                                        'new': 0.0,
                                        'diff': diff,
                                        'reason': "Elimination",
                                        'context': context
                                    })

            # End of game processing
            if game_winner is not None:
                winners.append(game_winner)
                win_types.append(game_win_type)
                game_lengths.append(moves_in_game)

    # --- Reporting ---
    print("\n" + "="*40)
    print(f"ANALYSIS REPORT: {len(winners)} Games Completed")
    print("="*40)
    
    if not winners:
        print("No valid games found.")
        return

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
    print("\n⏱️ Game Lengths (Moves):")
    if game_lengths:
        print(f"  Average: {statistics.mean(game_lengths):.1f}")
        print(f"  Median:  {statistics.median(game_lengths):.1f}")
        print(f"  Min:     {min(game_lengths)}")
        print(f"  Max:     {max(game_lengths)}")
        
    # 5. Blunders
    print(f"\n📉 Significant Winrate Shifts (Possible Blunders > {DROP_THRESHOLD}%):")
    if not blunders:
        print("  None detected.")
    else:
        for b in blunders:
            print(f"  Seed {b['seed']} Move {b['move_idx']} (P{b['player']}): {b['old']:.1f}% -> {b['new']:.1f}% ({b['diff']:.1f}%) [{b['reason']}]")
            print(f"    Context: {', '.join(b['context'])}")

    # 6. Confidence
    print("\n🤖 AI Confidence (Average Estimated Winrate):")
    for p in [1, 2, 3]:
        if est_winrates[p]:
            avg_wr = statistics.mean(est_winrates[p])
            print(f"  Player {p}: {avg_wr:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_log.py <log_file_pattern>")
        sys.exit(1)
    
    # Handle wildcards if the shell didn't expand them (e.g. Windows, or quoted)
    files = []
    for arg in sys.argv[1:]:
        expanded = glob.glob(arg)
        if expanded:
            files.extend(expanded)
        else:
            files.append(arg)
            
    analyze_squava_logs(files)
