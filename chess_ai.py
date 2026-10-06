"""
chess_ai.py - Heuristic Search AI for Chess
Academic project: demonstrates Depth-First Search (DFS) + Minimax with a
custom heuristic evaluation function.
"""
import chess

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

CHECKMATE_SCORE = 100_000

def evaluate_material(board):
    """Calculates pure material balance in centipawns (+ White, - Black)."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            val = PIECE_VALUES[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    return score

def evaluate_position(board, mode="material"):
    """Static evaluation baseline."""
    if board.is_checkmate():
        return -CHECKMATE_SCORE if board.turn == chess.WHITE else CHECKMATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return evaluate_material(board)
