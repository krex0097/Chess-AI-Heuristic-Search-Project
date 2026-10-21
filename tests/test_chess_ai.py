"""
test_chess_ai.py - Unit Test Suite for Chess AI Heuristic Search
================================================================
Academic Project: Comprehensive test suite validating:
- Heuristic evaluation function (material, PST, mobility, center, king safety, checkmate)
- Minimax depth-first search
- Alpha-Beta pruning correctness and pruning efficiency
- Tactical problem solving (Mate in 1 as White, Mate in 1 as Black, Stalemate avoidance)
- Non-destructive board simulation (push/pop integrity)
- Search diagnostics and performance statistics
"""

import unittest
import chess

import chess_ai as ai


class TestChessHeuristicEvaluation(unittest.TestCase):
    """Tests the static heuristic evaluation function and its individual components."""

    def test_starting_position_symmetry(self):
        """In the standard chess starting position, material balance must be exactly 0."""
        board = chess.Board()
        mat = ai.evaluate_material(board)
        self.assertEqual(mat, 0, f"Expected initial material balance 0, got {mat}")

        breakdown = ai.evaluate_position_breakdown(board, mode="full")
        self.assertFalse(breakdown["is_terminal"])
        self.assertEqual(breakdown["material"], 0)
        # Position is mirror symmetric initially
        self.assertEqual(breakdown["pst"], 0)

    def test_material_values(self):
        """Validates standard centipawn valuations for each piece type."""
        # Board with White queen, Black rook
        board = chess.Board("8/8/8/4Q3/8/8/4r3/k6K w - - 0 1")
        # White has Queen (900), Black has Rook (500) -> Material diff +400
        mat = ai.evaluate_material(board)
        self.assertEqual(mat, 400)

    def test_knight_centralization_bonus(self):
        """Verifies that knights on central squares (d4, e4) score higher than rim knights (a1, h1)."""
        # White knight on d4 vs White knight on a1
        board_center = chess.Board("8/8/8/8/3N4/8/8/k6K w - - 0 1")
        board_rim    = chess.Board("8/8/8/8/8/8/8/N5kK w - - 0 1")

        pst_center = ai.evaluate_pst(board_center)
        pst_rim    = ai.evaluate_pst(board_rim)
        self.assertGreater(pst_center, pst_rim, "Centralized knight should receive higher PST bonus than rim knight")

    def test_checkmate_evaluation(self):
        """Verifies that checkmated positions return +/- 100,000."""
        # Fool's mate position (White is checkmated by Black Queen on h4)
        board_white_mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(board_white_mated.is_checkmate())
        score = ai.evaluate_position(board_white_mated)
        self.assertEqual(score, -ai.CHECKMATE_SCORE, "White mated must evaluate to -100,000")

        # Scholar's mate position (Black is checkmated by White Queen on f7)
        board_black_mated = chess.Board("r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        self.assertTrue(board_black_mated.is_checkmate())
        score_black = ai.evaluate_position(board_black_mated)
        self.assertEqual(score_black, ai.CHECKMATE_SCORE, "Black mated must evaluate to +100,000")

    def test_stalemate_evaluation(self):
        """Verifies that stalemate returns exactly 0."""
        # Famous stalemate position: Black king on a8, White king on c7, White queen on c8 is mate, but queen on b6 is stalemate
        board_stalemate = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
        # Black has no legal moves and is not in check
        self.assertTrue(board_stalemate.is_stalemate())
        score = ai.evaluate_position(board_stalemate)
        self.assertEqual(score, 0, "Stalemate must evaluate to 0 (draw)")

    def test_baseline_vs_full_mode(self):
        """Verifies baseline mode only includes material while full mode adds positional factors."""
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        bd_base = ai.evaluate_position_breakdown(board, mode="material")
        bd_full = ai.evaluate_position_breakdown(board, mode="full")

        self.assertEqual(bd_base["pst"], 0)
        self.assertEqual(bd_base["center"], 0)
        self.assertEqual(bd_base["mobility"], 0)
        # In full mode, 1. e4 controls center and opens diagonals
        self.assertNotEqual(bd_full["total"], bd_base["total"])


class TestChessSearchAlgorithms(unittest.TestCase):
    """Tests pure Minimax and Alpha-Beta search routines."""

    def test_board_state_preservation(self):
        """Confirms that searching hypothetical moves does NOT permanently alter the board state."""
        board = chess.Board()
        initial_fen = board.fen()

        stats = ai.find_best_move(board, depth=2, algorithm="alphabeta")

        self.assertEqual(board.fen(), initial_fen, "Search must pop all hypothetical moves, preserving FEN")
        self.assertIsNotNone(stats.best_move)
        self.assertIn(stats.best_move, board.legal_moves)

    def test_solve_mate_in_one_white(self):
        """AI must detect Qxf7# (Scholar's mate threat) when playing as White."""
        # White: Q on f3, B on c4. Black: king on e8, f7 is weak.
        board = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
        stats = ai.find_best_move(board, depth=2)

        # Expected winning move is f3f7 (Qxf7#)
        self.assertIsNotNone(stats.best_move)
        self.assertEqual(stats.best_move.uci(), "f3f7", f"Expected f3f7 (Qxf7#), AI chose {stats.best_move.uci()}")
        self.assertGreaterEqual(stats.best_score, ai.CHECKMATE_SCORE)

    def test_solve_mate_in_one_black(self):
        """AI must detect Qh4# (Fool's mate defense) when playing as Black."""
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        stats = ai.find_best_move(board, depth=2)

        # Expected winning move is d8h4 (Qh4#)
        self.assertIsNotNone(stats.best_move)
        self.assertEqual(stats.best_move.uci(), "d8h4", f"Expected d8h4 (Qh4#), AI chose {stats.best_move.uci()}")
        self.assertLessEqual(stats.best_score, -ai.CHECKMATE_SCORE)

    def test_minimax_and_alphabeta_equivalence(self):
        """
        Theoretical guarantee: Pure Minimax and Alpha-Beta must produce the exact
        same best move and backed-up minimax score, but Alpha-Beta evaluates fewer nodes.
        """
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5")

        stats_pure = ai.find_best_move(board.copy(), depth=2, algorithm="minimax")
        stats_ab   = ai.find_best_move(board.copy(), depth=2, algorithm="alphabeta")

        self.assertEqual(stats_pure.best_score, stats_ab.best_score, "Minimax and Alpha-Beta scores must match")
        self.assertEqual(stats_pure.best_move, stats_ab.best_move, "Minimax and Alpha-Beta must choose identical moves")
        self.assertLessEqual(stats_ab.nodes_visited, stats_pure.nodes_visited,
                             "Alpha-Beta must visit fewer or equal nodes compared to pure Minimax")
        self.assertGreater(stats_ab.alpha_beta_cuts, 0, "Alpha-Beta must perform cutoffs on non-trivial positions")

    def test_search_depth_scaling(self):
        """Verifies that nodes visited grows exponentially with depth."""
        board = chess.Board()

        s1 = ai.find_best_move(board.copy(), depth=1, algorithm="alphabeta")
        s2 = ai.find_best_move(board.copy(), depth=2, algorithm="alphabeta")

        self.assertGreater(s2.nodes_visited, s1.nodes_visited)
        self.assertGreater(s1.elapsed_ms, 0)
        self.assertGreater(len(s2.top_moves), 0)

    def test_stalemate_avoidance(self):
        """When winning easily, the AI must not play a move that results in stalemate."""
        # White has King on b6 and Queen on d7. Black King on a8.
        # Moving Queen to c7 or b7 is mate. Moving Queen to a7 is capture/mate.
        # But moving King or Queen carelessly could produce stalemate.
        board = chess.Board("k7/3Q4/1K6/8/8/8/8/8 w - - 0 1")
        stats = ai.find_best_move(board, depth=2)

        board.push(stats.best_move)
        self.assertFalse(board.is_stalemate(), "AI should never choose an accidental stalemate when a forced win exists")


if __name__ == "__main__":
    unittest.main()

