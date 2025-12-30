import abc

# Constants
BOARD_SIZE = 8
WIN_LENGTH = 4
LOSE_LENGTH = 3

class Player(abc.ABC):
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol

    @abc.abstractmethod
    def get_move(self, board, forced_moves=None):
        """
        Get the move from the player.
        board: The current board state.
        forced_moves: A list of (r, c) tuples. If not None/Empty, the player MUST choose one of these.
        Returns: (r, c) tuple.
        """
        pass

    def __str__(self):
        return f"{self.name} ({self.symbol})"

class HumanPlayer(Player):
    def get_move(self, board, forced_moves=None):
        valid = False
        while not valid:
            prompt = f"{self.name} ({self.symbol}), enter your move (e.g., A1 or 0,0): "
            if forced_moves:
                # Convert forced moves to A1 format for display
                forced_str = [f"{chr(c + 65)}{r + 1}" for r, c in forced_moves]
                print(f"FORCED MOVE! You must block the next player. Valid moves: {', '.join(forced_str)}")
            
            user_input = input(prompt).strip().upper()
            
            try:
                r, c = self._parse_input(user_input)
                
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    print("Move out of bounds.")
                    continue
                
                if board[r][c] is not None:
                    print("Cell already occupied.")
                    continue
                    
                if forced_moves and (r, c) not in forced_moves:
                    # Check if the move is a winning move for us. 
                    # If we win, we don't need to block. 
                    # But the logic for that is usually handled by passing winning moves into 'forced_moves' 
                    # or handling it in the game loop.
                    # Based on my reasoning, winning moves should be allowed.
                    # However, strictly adhering to the passed forced_moves list is safer for this method.
                    # I will assume the game logic adds winning moves to forced_moves if that's the policy,
                    # OR I should check if it's a winning move here?
                    # Better design: Game class calculates valid_moves and passes them.
                    # But the interface says `forced_moves`.
                    # Let's reject if not in forced_moves.
                    print("Invalid move. You must block the opponent or win immediately.")
                    continue

                return (r, c)

            except ValueError:
                print("Invalid format. Use algebraic (A1) or coordinate (row,col).")
                
    def _parse_input(self, inp):
        # Algebraic: A1 -> (0, 0), H8 -> (7, 7)
        # Letter is Column, Number is Row.
        # Note: A=0, B=1... 
        # Number: 1=0, 2=1...
        
        # Check for coordinate format "r,c"
        if ',' in inp:
            parts = inp.split(',')
            return int(parts[0].strip()), int(parts[1].strip())
        
        # Algebraic
        if len(inp) < 2:
            raise ValueError
            
        col_char = inp[0]
        row_str = inp[1:]
        
        if not col_char.isalpha() or not row_str.isdigit():
             # maybe user typed "1A" or something
             raise ValueError

        col = ord(col_char) - ord('A')
        row = int(row_str) - 1
        return row, col

class SquavaGame:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.players = []
        self.eliminated_players = [] # Keep track if needed, though stone persistence handles most.
        self.turn_idx = 0

    def add_player(self, player):
        self.players.append(player)

    def print_board(self):
        # Print column headers
        header = "   " + " ".join([chr(ord('A') + i) for i in range(BOARD_SIZE)])
        print(header)
        for r in range(BOARD_SIZE):
            row_str = f"{r+1:2} "
            for c in range(BOARD_SIZE):
                cell = self.board[r][c]
                symbol = cell.symbol if cell else "."
                row_str += f"{symbol} "
            print(row_str)

    def check_sequence(self, r, c, dr, dc, length, symbol):
        """Check if there is a sequence of 'length' stones of 'symbol' starting at r,c in direction dr,dc."""
        # This checks a specific segment.
        # More useful: check if placing a stone at r,c completes a sequence.
        pass

    def count_consecutive(self, r, c, dr, dc, symbol):
        """Count consecutive stones of 'symbol' centered/including r,c in direction dr,dc."""
        count = 1
        # Forward
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] and self.board[nr][nc].symbol == symbol:
            count += 1
            nr += dr
            nc += dc
        
        # Backward
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr][nc] and self.board[nr][nc].symbol == symbol:
            count += 1
            nr -= dr
            nc -= dc
            
        return count

    def check_win_condition(self, r, c, player):
        """Returns True if placing stone at r,c for player creates 4 in a row."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self.count_consecutive(r, c, dr, dc, player.symbol) >= WIN_LENGTH:
                return True
        return False

    def check_lose_condition(self, r, c, player):
        """Returns True if placing stone at r,c for player creates 3 in a row."""
        # Note: If it creates 4, it is a WIN, not a lose (unless rule says otherwise). 
        # Rule: "If you make 4 in a row and 3 in a row simultaneously you still win."
        # So check win first.
        # This function strictly checks for >= 3. 
        # Logic in game loop should be: if check_win: win. elif check_lose: lose.
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = self.count_consecutive(r, c, dr, dc, player.symbol)
            if count >= LOSE_LENGTH:
                return True # This might be >= 4 too, but that's handled by check_win priority.
        return False

    def get_winning_moves(self, player):
        """Find all empty spots where 'player' could play to win immediately."""
        winning_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] is None:
                    # Temporarily place? Or just use count logic assuming placement?
                    # Need to simulate placement or adjust count logic.
                    # count_consecutive assumes the stone is ALREADY there if we pass r,c.
                    # So we need to temporarily place.
                    self.board[r][c] = player
                    if self.check_win_condition(r, c, player):
                        winning_moves.append((r, c))
                    self.board[r][c] = None
        return winning_moves

    def get_valid_moves(self, current_player, next_player):
        """
        Determine valid moves for current_player, considering forced blocks.
        """
        # Check if next_player has winning moves (threats)
        threats = self.get_winning_moves(next_player)
        
        # Also check if current_player has winning moves.
        # If current_player can win, they can do that instead of blocking.
        my_wins = self.get_winning_moves(current_player)
        
        if not threats:
            return None # No restrictions (other than empty cells)
        
        # If there are threats, valid moves are: Threats (blocking) + My Wins
        # We return this specific set.
        valid = set(threats) | set(my_wins)
        return list(valid)

    def is_board_full(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] is None:
                    return False
        return True

    def run(self):
        print("Starting 3-Player Squava!")
        print("Board Size: 8x8")
        print("Rules: 4-in-a-row wins. 3-in-a-row loses.")
        
        while True:
            if len(self.players) == 0: # Should not happen unless all elim
                 print("All players eliminated? Draw.")
                 break
            if len(self.players) == 1:
                print(f"{self.players[0].name} wins as the last player standing!")
                break
                
            current_player = self.players[self.turn_idx]
            next_player_idx = (self.turn_idx + 1) % len(self.players)
            next_player = self.players[next_player_idx]
            
            self.print_board()
            print(f"Turn: {current_player.name} ({current_player.symbol})")
            
            forced_moves = self.get_valid_moves(current_player, next_player)
            
            move = current_player.get_move(self.board, forced_moves)
            r, c = move
            
            # Place stone
            self.board[r][c] = current_player
            
            # Check Win
            if self.check_win_condition(r, c, current_player):
                self.print_board()
                print(f"!!! {current_player.name} wins with 4 in a row! !!!")
                return # Game Over
            
            # Check Lose
            if self.check_lose_condition(r, c, current_player):
                print(f"Oops! {current_player.name} made 3 in a row and is eliminated!")
                # Remove player
                self.players.pop(self.turn_idx)
                # Do not increment turn_idx, as the next player shifts into this slot.
                # However, ensure index wraps if we were at the end.
                if self.turn_idx >= len(self.players):
                    self.turn_idx = 0
                
                # Check bounds for remaining players
                if self.is_board_full():
                     self.print_board()
                     print("Board full! Game is a Draw between remaining players.")
                     return
                continue
            
            if self.is_board_full():
                self.print_board()
                print("Board full! Game is a Draw.")
                return

            self.turn_idx = (self.turn_idx + 1) % len(self.players)

if __name__ == "__main__":
    game = SquavaGame()
    game.add_player(HumanPlayer("Player 1", "X"))
    game.add_player(HumanPlayer("Player 2", "O"))
    game.add_player(HumanPlayer("Player 3", "Z"))
    game.run()
