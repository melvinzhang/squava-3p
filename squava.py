import abc
import argparse
import random
import math
import copy
import time

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
                    print("Invalid move. You must block the opponent or win immediately.")
                    continue

                return (r, c)

            except ValueError:
                print("Invalid format. Use algebraic (A1) or coordinate (row,col).")
                
    def _parse_input(self, inp):
        # Algebraic: A1 -> (0, 0), H8 -> (7, 7)
        # Letter is Column, Number is Row.
        if ',' in inp:
            parts = inp.split(',')
            return int(parts[0].strip()), int(parts[1].strip())
        
        if len(inp) < 2:
            raise ValueError
            
        col_char = inp[0]
        row_str = inp[1:]
        
        if not col_char.isalpha() or not row_str.isdigit():
             raise ValueError

        col = ord(col_char) - ord('A')
        row = int(row_str) - 1
        return row, col

class MCTSNode:
    def __init__(self, board, parent, player_to_move, remaining_players, winner=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0 # for player_to_move
        self.player_to_move = player_to_move # The player whose turn it is in this state
        self.remaining_players = remaining_players # List of players still in game
        self.winner = winner
        self.untried_moves = self.get_possible_moves()
        
    def get_possible_moves(self):
        if self.winner or self.player_to_move is None:
            return []
            
        # Calculate valid moves based on game rules (forced moves)
        # We need next player to calculate forced moves.
        # But this node's state is "before player_to_move moves".
        # So we need to know who is AFTER player_to_move.
        
        if not self.remaining_players:
            return []
            
        current_idx = -1
        for i, p in enumerate(self.remaining_players):
            if p.symbol == self.player_to_move.symbol:
                current_idx = i
                break
        
        if current_idx == -1: return [] # Player eliminated?
        
        next_idx = (current_idx + 1) % len(self.remaining_players)
        next_player = self.remaining_players[next_idx]
        
        # We need a static helper for logic to avoid instantiating full Game
        valid_moves = SquavaGame.static_get_valid_moves(self.board, self.player_to_move, next_player)
        
        if valid_moves is None:
             # All empty cells
             valid_moves = []
             for r in range(BOARD_SIZE):
                 for c in range(BOARD_SIZE):
                     if self.board[r][c] is None:
                         valid_moves.append((r, c))
        return valid_moves

    def uct_select_child(self):
        # UCB1
        log_visits = math.log(self.visits)
        best_score = -float('inf')
        best_child = None
        
        for move, child in self.children.items():
            # exploitation: child.wins / child.visits (Win rate from perspective of child.player_prev)
            # Wait, standard MCTS: child.wins is wins for the player who MOVED to get to child?
            # Or wins for the player whose turn it is at child?
            # Usually we store "value for the player who just moved".
            # Let's say: value is "score for the player who made the move leading to this node".
            
            # If child represents state after P1 moved.
            # value should be P1's win rate.
            # But self.player_to_move is P1.
            # So we want to maximize P1's win rate.
            
            if child.visits == 0:
                 score = float('inf')
            else:
                 # child.wins is relative to the player who made the move (self.player_to_move)
                 win_rate = child.wins / child.visits
                 explore = math.sqrt(2 * log_visits / child.visits)
                 score = win_rate + explore
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def add_child(self, move, state):
        child = MCTSNode(state['board'], self, state['next_player'], state['remaining_players'], state['winner'])
        self.children[move] = child
        return child
        
    def update(self, result):
        self.visits += 1
        # result: {player_symbol: score}
        # score: 1 for win, 0 for loss/draw
        # self.player_to_move is the one who is ABOUT to move from this state.
        # But this node is a child of a previous node. 
        # Actually, simpler:
        # Node stores 'wins'.
        # For MCTS with >2 players, 'wins' is usually specific to the player who just moved.
        # Let's verify standard UCT.
        # Parent (P1 to move) -> Child (P2 to move).
        # We want to pick Child that is good for P1.
        # Child's stats should reflect P1's success.
        # So we accumulate wins for the player who moved to GET here.
        # Who moved to get to 'self'? self.parent.player_to_move.
        
        if self.parent:
             mover = self.parent.player_to_move
             if mover.symbol in result and result[mover.symbol] == 1:
                 self.wins += 1
        # Root node doesn't need wins/visits really for selection, but for stats.

class MCTSPlayer(Player):
    def __init__(self, name, symbol, iterations=1000):
        super().__init__(name, symbol)
        self.iterations = iterations

    def get_move(self, board, forced_moves=None):
        # We need to reconstruct the players list for the simulation
        # This is tricky because `board` doesn't strictly have the list of players.
        # But the `SquavaGame` instance passed `self.board`.
        # We only get `board`. We don't get the list of active players in the interface.
        # We need to infer active players? Or change the interface.
        # The `board` contains Player objects in the cells.
        # We can scan the board to find active symbols, but we don't know the turn order or who is eliminated just from the board if they have 0 stones?
        # Actually, eliminated players keep their stones.
        # So we cannot infer active players just from board.
        # I should update `get_move` to receive `game_state` or just assume standard 3 players if not provided?
        # Or, since I am editing `SquavaGame`, I can pass `game` object instead of board?
        # Or pass `remaining_players`.
        
        # Let's Hack: pass `game` instance in `get_move`? 
        # Or cleaner: `MCTSPlayer` takes a reference to the `game` object on init?
        # But `game` is created after players.
        # Let's modify `get_move` signature in `SquavaGame` to pass `self` (the game) or `remaining_players`.
        # I will modify `get_move` to take `game_context`.
        
        # For now, I'll assume standard 3 players if I can't get context, 
        # but better to fix the interface.
        # I will modify SquavaGame to pass `self` as context.
        pass

    def get_move_with_context(self, board, forced_moves, players, turn_idx):
        root = MCTSNode(self._copy_board(board), None, players[turn_idx], players)
        
        # If forced moves, we only consider those.
        # But MCTS should explore within those.
        # MCTSNode `get_possible_moves` logic handles this if we implement `static_get_valid_moves` correctly.
        # However, we can also prune the root:
        if forced_moves:
             root.untried_moves = forced_moves # Override
        
        start_time = time.time()
        for _ in range(self.iterations):
            node = root
            
            # Selection
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
            
            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state = self._simulate_step(node.board, node.remaining_players, node.player_to_move, move)
                node = node.add_child(move, state)
                
            # Simulation
            if node.winner:
                result = {node.winner.symbol: 1}
            else:
                result = self._run_simulation(node.board, node.remaining_players, node.player_to_move)
            
            # Backprop
            while node:
                node.update(result)
                node = node.parent
        
        # Select best move (most visited)
        if not root.children:
             # Should not happen unless no moves
             return (0,0)
             
        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_move

    def _copy_board(self, board):
        return [row[:] for row in board]

    def _simulate_step(self, board, players, current_player, move):
        # Returns new state dict: board, next_player, remaining_players
        r, c = move
        new_board = self._copy_board(board)
        new_board[r][c] = current_player
        
        new_players = list(players)
        
        # Logic for win/loss
        # If win: game over
        # If lose: remove player
        
        # We need to find current_player index in new_players
        try:
            p_idx = [p.symbol for p in new_players].index(current_player.symbol)
        except ValueError:
             # Should not happen
             p_idx = 0

        next_player = None
        
        # Check Win
        if SquavaGame.static_check_win_condition(new_board, r, c, current_player):
            # Game Over - Winner is current_player
            return {'board': new_board, 'next_player': None, 'remaining_players': new_players, 'winner': current_player}
            
        # Check Lose
        if SquavaGame.static_check_lose_condition(new_board, r, c, current_player):
             # Eliminate current player
             new_players.pop(p_idx)
             if len(new_players) == 1:
                 # Winner is the survivor
                 return {'board': new_board, 'next_player': None, 'remaining_players': new_players, 'winner': new_players[0]}
             
             # Turn passes to the player who was at p_idx (now shifted) or wrap
             if p_idx >= len(new_players):
                 p_idx = 0
             next_player = new_players[p_idx]
        else:
             # Normal turn pass
             next_idx = (p_idx + 1) % len(new_players)
             next_player = new_players[next_idx]
             
        return {'board': new_board, 'next_player': next_player, 'remaining_players': new_players, 'winner': None}

    def _run_simulation(self, board, players, current_player):
        # Play random moves until game ends
        sim_board = self._copy_board(board)
        sim_players = list(players)
        
        # If passed state has winner or single player, return result immediately
        if len(sim_players) == 1:
             return {sim_players[0].symbol: 1}
        
        # We need to know who is 'next' to start the loop
        # The node passed 'current_player' as the one whose turn it WAS.
        # But 'state' passed to child is 'after move'.
        # Wait, `_simulate_step` returns `next_player`.
        # But `add_child` uses that state.
        # So `node.player_to_move` in the child IS the next player.
        
        curr = current_player
        
        while True:
            # Check if game over (already checked in step, but we are loop)
            if len(sim_players) == 1:
                return {sim_players[0].symbol: 1}
                
            # Find next player to move
            # In simulation loop, 'curr' is the one to move
            
            # Get valid moves (random)
            # Optimization: don't calculate all forced moves? 
            # Or simplified simulation?
            # For correctness, we should respect forced moves.
            
            # Identify next player (target of blocks)
            p_idx = -1
            for i, p in enumerate(sim_players):
                if p.symbol == curr.symbol:
                    p_idx = i
                    break
            
            next_p = sim_players[(p_idx + 1) % len(sim_players)]
            
            valid_moves = SquavaGame.static_get_valid_moves(sim_board, curr, next_p)
            if valid_moves is None:
                # Get all empty
                valid_moves = []
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        if sim_board[r][c] is None:
                            valid_moves.append((r, c))
            
            if not valid_moves:
                # Draw
                return {} 
            
            move = random.choice(valid_moves)
            step_res = self._simulate_step(sim_board, sim_players, curr, move)
            
            sim_board = step_res['board']
            sim_players = step_res['remaining_players']
            
            if step_res['winner']:
                return {step_res['winner'].symbol: 1}
            
            if not sim_players: # Draw?
                 return {}
            
            curr = step_res['next_player']
            if curr is None: # Should be winner handled
                # If winner was found, we returned.
                # If eliminated, simulate step returns next player.
                break

class SquavaGame:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.players = []
        self.eliminated_players = [] 
        self.turn_idx = 0

    def add_player(self, player):
        self.players.append(player)

    def print_board(self):
        header = "   " + " ".join([chr(ord('A') + i) for i in range(BOARD_SIZE)])
        print(header)
        for r in range(BOARD_SIZE):
            row_str = f"{r+1:2} "
            for c in range(BOARD_SIZE):
                cell = self.board[r][c]
                symbol = cell.symbol if cell else "."
                row_str += f"{symbol} "
            print(row_str)

    @staticmethod
    def count_consecutive(board, r, c, dr, dc, symbol):
        count = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] and board[nr][nc].symbol == symbol:
            count += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] and board[nr][nc].symbol == symbol:
            count += 1
            nr -= dr
            nc -= dc
        return count

    @staticmethod
    def static_check_win_condition(board, r, c, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if SquavaGame.count_consecutive(board, r, c, dr, dc, player.symbol) >= WIN_LENGTH:
                return True
        return False

    @staticmethod
    def static_check_lose_condition(board, r, c, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if SquavaGame.count_consecutive(board, r, c, dr, dc, player.symbol) >= LOSE_LENGTH:
                return True
        return False
        
    def check_win_condition(self, r, c, player):
        return self.static_check_win_condition(self.board, r, c, player)
        
    def check_lose_condition(self, r, c, player):
        return self.static_check_lose_condition(self.board, r, c, player)

    @staticmethod
    def static_get_winning_moves(board, player):
        winning_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] is None:
                    board[r][c] = player
                    if SquavaGame.static_check_win_condition(board, r, c, player):
                        winning_moves.append((r, c))
                    board[r][c] = None
        return winning_moves

    @staticmethod
    def static_get_valid_moves(board, current_player, next_player):
        threats = SquavaGame.static_get_winning_moves(board, next_player)
        my_wins = SquavaGame.static_get_winning_moves(board, current_player)
        
        if not threats:
            return None
        valid = set(threats) | set(my_wins)
        return list(valid)

    def get_valid_moves(self, current_player, next_player):
        return self.static_get_valid_moves(self.board, current_player, next_player)

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
            if len(self.players) == 0:
                 print("All players eliminated? Draw.")
                 break
            if len(self.players) == 1:
                print(f"{self.players[0].name} wins as the last player standing!")
                break
                
            current_player = self.players[self.turn_idx]
            next_player_idx = (self.turn_idx + 1) % len(self.players)
            next_player = self.players[next_player_idx]
            
            # self.print_board() # Optional: Don't print every step if MCTS vs MCTS
            print(f"Turn: {current_player.name} ({current_player.symbol})")
            
            forced_moves = self.get_valid_moves(current_player, next_player)
            
            # Pass context to MCTSPlayer if needed
            if isinstance(current_player, MCTSPlayer):
                print(f"{current_player.name} is thinking...")
                move = current_player.get_move_with_context(self.board, forced_moves, self.players, self.turn_idx)
                print(f"{current_player.name} chooses {chr(move[1]+65)}{move[0]+1}")
            else:
                self.print_board()
                move = current_player.get_move(self.board, forced_moves)
            
            r, c = move
            self.board[r][c] = current_player
            
            if self.check_win_condition(r, c, current_player):
                self.print_board()
                print(f"!!! {current_player.name} wins with 4 in a row! !!!")
                return 
            
            if self.check_lose_condition(r, c, current_player):
                print(f"Oops! {current_player.name} made 3 in a row and is eliminated!")
                self.players.pop(self.turn_idx)
                if self.turn_idx >= len(self.players):
                    self.turn_idx = 0
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
    parser = argparse.ArgumentParser(description="Squava 3-Player Game")
    parser.add_argument("--p1", type=str, choices=["human", "mcts"], default="human", help="Player 1 type")
    parser.add_argument("--p2", type=str, choices=["human", "mcts"], default="human", help="Player 2 type")
    parser.add_argument("--p3", type=str, choices=["human", "mcts"], default="human", help="Player 3 type")
    parser.add_argument("--iterations", type=int, default=1000, help="MCTS iterations")
    
    args = parser.parse_args()
    
    game = SquavaGame()
    
    def create_player(type_str, name, symbol):
        if type_str == "human":
            return HumanPlayer(name, symbol)
        else:
            return MCTSPlayer(name, symbol, iterations=args.iterations)

    game.add_player(create_player(args.p1, "Player 1", "X"))
    game.add_player(create_player(args.p2, "Player 2", "O"))
    game.add_player(create_player(args.p3, "Player 3", "Z"))
    
    game.run()