"""
chess_ai.py - Artificial Intelligence Heuristic Search for Chess
================================================================
Academic Project: Demonstrates Depth-First Search (DFS) Game Tree Traversal,
Depth-Limited Minimax Decision Making, Alpha-Beta Pruning, and Custom
Heuristic Evaluation Functions.

Key Algorithmic Concepts:
-------------------------
1. Depth-First Search (DFS):
   Recursively traverses each candidate branch of the chess game tree to the
   configured depth limit before backtracking. Space complexity is O(b * d).

2. Minimax Decision Rule:
   Alternates between the Maximizing player (White, seeking higher scores)
   and the Minimizing player (Black, seeking lower scores).
   Terminal states and depth-limited leaf nodes are scored by the heuristic.

3. Alpha-Beta Pruning:
   An optimization over pure Minimax that maintains two bounds:
   - alpha: the minimum score the maximizing player is assured of.
   - beta:  the maximum score the minimizing player is assured of.
   Branches with beta <= alpha are pruned, avoiding unnecessary evaluation
   without altering the chosen minimax value.

4. Heuristic Evaluation Function:
   Evaluates quiescent/leaf positions when depth limit is reached:
   - Material balance (Pawn: 100, Knight: 320, Bishop: 330, Rook: 500, Queen: 900)
   - Piece-Square Tables (PST) for positional placement
   - Center control (occupation and attack of d4, e4, d5, e5 + extended center)
   - Piece mobility (legal attack/movement scope)
   - King safety (pawn shield and open file exposure)
   - Baseline mode: evaluates material balance alone for academic comparison.

Score Convention:
-----------------
Positive scores favor White; negative scores favor Black; 0 is equal.
Checkmate scores (+/- 100,000) dominate all positional factors.
Depth is factored into mate scores to prefer faster mates and delay losses.
"""

import time
import argparse
import chess

# ---------------------------------------------------------------------------
#  Material Values (Centipawns: 1 pawn = 100 cp)
# ---------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,    # King evaluated via checkmate/king-safety terms
}

CHECKMATE_SCORE = 100_000   # Dominates any positional sum

# Center squares definitions
INNER_CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
OUTER_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}

# ---------------------------------------------------------------------------
#  Piece-Square Tables (PST)
#  Values in centipawns. Defined from White's POV (rank 8 down to rank 1).
#  Mirrored symmetrically for Black along the rank axis.
# ---------------------------------------------------------------------------
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_PST_MIDGAME = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST_TABLES = {
    chess.PAWN:   PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK:   ROOK_PST,
    chess.QUEEN:  QUEEN_PST,
    chess.KING:   KING_PST_MIDGAME,
}


def get_pst_bonus(piece_type, square, color):
    """
    Look up positional bonus for piece_type at square.
    Mirrors ranks for Black so both sides use the same strategic priorities.
    """
    table = PST_TABLES.get(piece_type)
    if table is None:
        return 0
    if color == chess.WHITE:
        idx = (7 - chess.square_rank(square)) * 8 + chess.square_file(square)
    else:
        idx = chess.square_rank(square) * 8 + chess.square_file(square)
    return table[idx]



def evaluate_material(board):
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            val = PIECE_VALUES[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    return score

def evaluate_pst(board):
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = get_pst_bonus(piece.piece_type, sq, piece.color)
            score += val if piece.color == chess.WHITE else -val
    return score

def evaluate_position(board, mode="full"):
    if board.is_checkmate():
        return -CHECKMATE_SCORE if board.turn == chess.WHITE else CHECKMATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return evaluate_material(board) + evaluate_pst(board)
