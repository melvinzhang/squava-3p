import unittest
from squava import SquavaGame, Player, BOARD_SIZE

class ScriptedPlayer(Player):
    def __init__(self, name, symbol, moves):
        super().__init__(name, symbol)
        self.moves = moves # list of (r, c)
        self.move_idx = 0

    def get_move(self, board, forced_moves=None):
        if self.move_idx < len(self.moves):
            move = self.moves[self.move_idx]
            self.move_idx += 1
            
            # Check forced logic compliance for test verification
            if forced_moves:
                if move not in forced_moves:
                    # In a real game this is blocked. In test, we might want to assert fail.
                    # But here let's just return the move and let the game handle it (if we enforce it in game loop?)
                    # Wait, SquavaGame class doesn't Enforce forced moves inside 'run' strictly? 
                    # Ah, I moved enforcement to HumanPlayer.get_move.
                    # The Game class calculates valid moves but doesn't throw if player ignores them?
                    # Let's check SquavaGame.run code.
                    pass
            return move
        return (0, 0) # Fallback

class TestSquavaLogic(unittest.TestCase):
    def setUp(self):
        self.game = SquavaGame()

    def test_win_horizontal(self):
        # P1 plays (0,0), (0,1), (0,2), (0,3)
        self.game.board[0][0] = self.game.board[0][1] = self.game.board[0][2] = self.game.board[0][3] = None
        p1 = ScriptedPlayer("P1", "X", [])
        # Place 3
        self.game.board[0][0] = p1
        self.game.board[0][1] = p1
        self.game.board[0][2] = p1
        # Check win on 4th
        self.assertTrue(self.game.check_win_condition(0, 3, p1)) # Note: checks if placing at 0,3 wins. 
        # But wait, check_win_condition counts consecutive including the one at r,c.
        # But assumes board[r][c] is set? 
        # squava.py: 
        # count = 1
        # forward ... board[nr][nc] == symbol
        # So yes, it counts neighbors. It counts the '1' for the center implicitly (start count=1).
        # It does NOT check if board[r][c] itself is the symbol. It assumes we are "placing" it there or testing a hypothesis.
        # Wait, let's look at count_consecutive implementation again.
        
    def test_count_consecutive_logic(self):
        # squava.py logic:
        # count = 1
        # loop forward: checks board[nr][nc].
        # loop backward: checks board[nr][nc].
        # It does NOT check board[r][c]. It assumes the stone at (r,c) contributes +1.
        # This is correct for "check if placing here wins".
        
        p1 = ScriptedPlayer("P1", "X", [])
        self.game.board[0][0] = p1
        self.game.board[0][1] = p1
        self.game.board[0][2] = p1
        
        # Check if placing at 0,3 wins
        # (0,3) neighbors (0,2) (0,1) (0,0) in direction (0, -1)
        self.assertTrue(self.game.check_win_condition(0, 3, p1))
        
    def test_lose_condition(self):
        p1 = ScriptedPlayer("P1", "X", [])
        self.game.board[0][0] = p1
        self.game.board[0][1] = p1
        # Placing at 0,2 makes 3 in a row -> Lose
        self.assertTrue(self.game.check_lose_condition(0, 2, p1))
        # But check win should be false
        self.assertFalse(self.game.check_win_condition(0, 2, p1))

    def test_simultaneous_win_lose(self):
        # Setup: X X _ X X
        # Placing in middle makes 5 (Win) but implies 3 (Lose).
        # Win should be True.
        p1 = ScriptedPlayer("P1", "X", [])
        self.game.board[0][0] = p1
        self.game.board[0][1] = p1
        # 0,2 is empty
        self.game.board[0][3] = p1
        self.game.board[0][4] = p1
        
        # Placing at 0,2 makes 5 consecutive.
        self.assertTrue(self.game.check_win_condition(0, 2, p1))
        # It also triggers check_lose (>=3)
        self.assertTrue(self.game.check_lose_condition(0, 2, p1))
        
        # In actual run loop, win is checked first.
        
    def test_forced_moves(self):
        # P1 (X), P2 (O), P3 (Z)
        # Board:
        # X X X _ (P2 to move)
        # If P2 moves, P1 is not next. P3 is next.
        # Wait, forced move is: "If the player immediately AFTER you can win".
        # Order: P1 -> P2 -> P3 -> P1
        # Current: P1. Next: P2.
        # If P2 has a winning spot, P1 must block.
        
        p1 = ScriptedPlayer("P1", "X", [])
        p2 = ScriptedPlayer("P2", "O", [])
        
        # Setup P2 to have 3 in a row
        self.game.board[1][0] = p2
        self.game.board[1][1] = p2
        self.game.board[1][2] = p2
        # P2 wins at (1,3).
        
        # Check threats for P2
        threats = self.game.get_winning_moves(p2)
        self.assertIn((1, 3), threats)
        
        # Now check P1's valid moves.
        # P1 must block (1,3).
        valid = self.game.get_valid_moves(p1, p2)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0], (1, 3))
        
    def test_forced_moves_exemption(self):
        # "If the second player after you can win ... you needn't stop him"
        # Current: P1. Next: P2. NextNext: P3.
        # If P3 has win, P1 ignores it.
        
        p1 = ScriptedPlayer("P1", "X", [])
        p2 = ScriptedPlayer("P2", "O", [])
        p3 = ScriptedPlayer("P3", "Z", [])
        
        # Setup P3 to have 3 in a row
        self.game.board[2][0] = p3
        self.game.board[2][1] = p3
        self.game.board[2][2] = p3
        
        # P3 wins at (2,3).
        
        # Current is P1. Next is P2.
        # P2 has no threats.
        valid = self.game.get_valid_moves(p1, p2)
        
        # Should be None (all moves allowed)
        self.assertIsNone(valid)

if __name__ == '__main__':
    unittest.main()
