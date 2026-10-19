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


# ---------------------------------------------------------------------------
#  Evaluation Components
# ---------------------------------------------------------------------------

def evaluate_material(board):
    """Calculates pure material balance in centipawns (+ White, - Black)."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            val = PIECE_VALUES[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    return score


def evaluate_pst(board):
    """Calculates Piece-Square Table positional bonuses (+ White, - Black)."""
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = get_pst_bonus(piece.piece_type, sq, piece.color)
            score += val if piece.color == chess.WHITE else -val
    return score


def evaluate_center_control(board):
    """
    Measures center control:
    - Inner center squares (d4, e4, d5, e5): 15 cp each
    - Outer center squares: 4 cp each
    Accounts for both physical piece occupation and attack coverage.
    """
    score = 0
    # Occupation
    for sq in INNER_CENTER:
        p = board.piece_at(sq)
        if p:
            score += 15 if p.color == chess.WHITE else -15
    for sq in OUTER_CENTER:
        p = board.piece_at(sq)
        if p:
            score += 4 if p.color == chess.WHITE else -4

    # Attack pressure
    for sq in INNER_CENTER:
        w_att = len(board.attackers(chess.WHITE, sq))
        b_att = len(board.attackers(chess.BLACK, sq))
        score += (w_att - b_att) * 6

    return score


def evaluate_mobility(board):
    """
    Piece mobility bonus: counts the number of pseudo-legal moves available
    to each side. More mobile pieces signify better coordination and board influence.
    Reward ~2 centipawns per move difference.
    """
    # Temporarily check moves from current turn
    curr_turn = board.turn
    board.turn = chess.WHITE
    white_moves = board.legal_moves.count()
    board.turn = chess.BLACK
    black_moves = board.legal_moves.count()
    board.turn = curr_turn

    return (white_moves - black_moves) * 2


def evaluate_king_safety(board):
    """
    Evaluates king safety:
    - Pawn shield: checks pawns directly in front of the castled king (10 cp per shield pawn).
    - Open files near the king: penalizes rooks/queens aiming at exposed king files (-15 cp).
    """
    score = 0
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)

    if w_king_sq is not None:
        w_file = chess.square_file(w_king_sq)
        w_rank = chess.square_rank(w_king_sq)
        # Check shield pawns on rank 2/3 for White if castled (file 1-2 or 6-7)
        if w_rank <= 1 and (w_file <= 2 or w_file >= 5):
            shield_files = [f for f in (w_file - 1, w_file, w_file + 1) if 0 <= f <= 7]
            for f in shield_files:
                for r in (1, 2):
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                        score += 10

    if b_king_sq is not None:
        b_file = chess.square_file(b_king_sq)
        b_rank = chess.square_rank(b_king_sq)
        if b_rank >= 6 and (b_file <= 2 or b_file >= 5):
            shield_files = [f for f in (b_file - 1, b_file, b_file + 1) if 0 <= f <= 7]
            for f in shield_files:
                for r in (6, 5):
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                        score -= 10

    return score


def evaluate_position_breakdown(board, mode="full"):
    """
    Returns an academic breakdown dictionary of the heuristic evaluation.
    mode="full": Material + PST + Center + Mobility + King Safety
    mode="material": Pure material baseline (no positional terms)
    """
    # 1. Terminal conditions
    if board.is_checkmate():
        # The player whose turn it is has been checkmated
        if board.turn == chess.WHITE:
            return {
                "total": -CHECKMATE_SCORE,
                "material": 0,
                "pst": 0,
                "center": 0,
                "mobility": 0,
                "king_safety": 0,
                "is_terminal": True,
                "terminal_reason": "Black delivered checkmate",
            }
        else:
            return {
                "total": CHECKMATE_SCORE,
                "material": 0,
                "pst": 0,
                "center": 0,
                "mobility": 0,
                "king_safety": 0,
                "is_terminal": True,
                "terminal_reason": "White delivered checkmate",
            }

    if board.is_stalemate():
        return {
            "total": 0,
            "material": 0,
            "pst": 0,
            "center": 0,
            "mobility": 0,
            "king_safety": 0,
            "is_terminal": True,
            "terminal_reason": "Stalemate (draw)",
        }

    if board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return {
            "total": 0,
            "material": 0,
            "pst": 0,
            "center": 0,
            "mobility": 0,
            "king_safety": 0,
            "is_terminal": True,
            "terminal_reason": "Draw by chess rules",
        }

    # 2. Heuristic terms
    mat_score = evaluate_material(board)

    if mode == "material":
        return {
            "total": mat_score,
            "material": mat_score,
            "pst": 0,
            "center": 0,
            "mobility": 0,
            "king_safety": 0,
            "is_terminal": False,
            "terminal_reason": None,
        }

    pst_score = evaluate_pst(board)
    center_score = evaluate_center_control(board)
    mobility_score = evaluate_mobility(board)
    king_score = evaluate_king_safety(board)

    total = mat_score + pst_score + center_score + mobility_score + king_score

    return {
        "total": total,
        "material": mat_score,
        "pst": pst_score,
        "center": center_score,
        "mobility": mobility_score,
        "king_safety": king_score,
        "is_terminal": False,
        "terminal_reason": None,
    }


def evaluate_position(board, mode="full"):
    """
    Static evaluation function. Returns integer score from White's POV.
    """
    breakdown = evaluate_position_breakdown(board, mode)
    return breakdown["total"]


# ---------------------------------------------------------------------------
#  Move Ordering (Optimizes Alpha-Beta Pruning Efficiency)
# ---------------------------------------------------------------------------

def _move_score(board, move):
    """
    Heuristic move ordering:
    - MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for captures.
    - Pawn promotions given high priority.
    - Castling given a small development bonus.
    """
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += 10 * PIECE_VALUES.get(victim.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0)
        else:
            score += 300  # e.g., en-passant
    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 900)
    if board.is_castling(move):
        score += 80
    return score


def generate_ordered_legal_moves(board):
    """Returns legal moves sorted with promising moves first to maximize alpha-beta cuts."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: _move_score(board, m), reverse=True)
    return moves


# ---------------------------------------------------------------------------
#  Search Statistics Accumulator
# ---------------------------------------------------------------------------

class SearchStats:
    """Collects observable metrics during the game tree search."""
    def __init__(self):
        self.nodes_visited = 0
        self.leaves_evaluated = 0
        self.alpha_beta_cuts = 0
        self.depth = 0
        self.elapsed_ms = 0.0
        self.best_move = None
        self.best_score = 0
        self.top_moves = []         # [(uci_str, san_str, minimax_score, static_score), ...]
        self.algorithm = "alphabeta"  # "alphabeta" or "minimax"
        self.heuristic_mode = "full"  # "full" or "material"
        self.explanation = ""
        self.breakdown = {}


# ---------------------------------------------------------------------------
#  Search Algorithms: DFS Pure Minimax & Minimax with Alpha-Beta Pruning
# ---------------------------------------------------------------------------

def pure_minimax(board, depth, is_maximizing, stats, heuristic_mode="full"):
    """
    Pure Depth-First Search (DFS) Minimax without pruning.
    Visits every reachable node up to depth limit. Ideal for teaching
    the foundational concept of minimax adversarial decision making.
    """
    stats.nodes_visited += 1

    # Base cases: depth limit reached or terminal state found
    if depth == 0 or board.is_game_over():
        stats.leaves_evaluated += 1
        raw = evaluate_position(board, heuristic_mode)
        # Factor in depth to prefer faster mates and delay forced losses
        if raw >= CHECKMATE_SCORE:
            return raw + depth
        if raw <= -CHECKMATE_SCORE:
            return raw - depth
        return raw

    moves = list(board.legal_moves)
    if not moves:
        stats.leaves_evaluated += 1
        return evaluate_position(board, heuristic_mode)

    if is_maximizing:
        best_val = -CHECKMATE_SCORE * 2
        for move in moves:
            board.push(move)
            val = pure_minimax(board, depth - 1, False, stats, heuristic_mode)
            board.pop()
            if val > best_val:
                best_val = val
        return best_val
    else:
        best_val = CHECKMATE_SCORE * 2
        for move in moves:
            board.push(move)
            val = pure_minimax(board, depth - 1, True, stats, heuristic_mode)
            board.pop()
            if val < best_val:
                best_val = val
        return best_val


def alphabeta_search(board, depth, alpha, beta, is_maximizing, stats, heuristic_mode="full"):
    """
    Depth-First Search (DFS) Minimax with Alpha-Beta Pruning.
    Yields the exact same decision and minimax value as pure minimax,
    while pruning branches that cannot affect the outcome.
    """
    stats.nodes_visited += 1

    # Base cases
    if depth == 0 or board.is_game_over():
        stats.leaves_evaluated += 1
        raw = evaluate_position(board, heuristic_mode)
        if raw >= CHECKMATE_SCORE:
            return raw + depth
        if raw <= -CHECKMATE_SCORE:
            return raw - depth
        return raw

    moves = generate_ordered_legal_moves(board)
    if not moves:
        stats.leaves_evaluated += 1
        return evaluate_position(board, heuristic_mode)

    if is_maximizing:
        best_val = -CHECKMATE_SCORE * 2
        for move in moves:
            board.push(move)
            val = alphabeta_search(board, depth - 1, alpha, beta, False, stats, heuristic_mode)
            board.pop()
            if val > best_val:
                best_val = val
            if best_val > alpha:
                alpha = best_val
            if beta <= alpha:
                stats.alpha_beta_cuts += 1
                break  # Beta cutoff
        return best_val 
    else:
        best_val = CHECKMATE_SCORE * 2
        for move in moves:
            board.push(move)
            val = alphabeta_search(board, depth - 1, alpha, beta, True, stats, heuristic_mode)
            board.pop()
            if val < best_val:
                best_val = val
            if best_val < beta:
                beta = best_val
            if beta <= alpha:
                stats.alpha_beta_cuts += 1
                break  # Alpha cutoff
        return best_val


# ---------------------------------------------------------------------------
#  Primary AI Decision Entry Point
# ---------------------------------------------------------------------------

def find_best_move(board, depth=3, algorithm="alphabeta", heuristic_mode="full"):
    """
    Searches the game tree from board position to depth plies and selects
    the optimal move using Minimax decision-making.

    Parameters:
    -----------
    board          : chess.Board instance
    depth          : search ply limit (>= 1)
    algorithm      : 'alphabeta' (pruned DFS) or 'minimax' (pure unpruned DFS)
    heuristic_mode : 'full' (enhanced positional) or 'material' (baseline)

    Returns:
    --------
    SearchStats containing chosen move, statistics, candidate rankings, and explanation.
    """
    stats = SearchStats()
    stats.depth = depth
    stats.algorithm = algorithm
    stats.heuristic_mode = heuristic_mode

    # White is maximizing (+), Black is minimizing (-)
    is_white = (board.turn == chess.WHITE)
    best_score = -CHECKMATE_SCORE * 2 if is_white else CHECKMATE_SCORE * 2
    best_move = None
    candidate_records = []

    alpha = -CHECKMATE_SCORE * 2
    beta  =  CHECKMATE_SCORE * 2

    start_time = time.perf_counter()
    legal_moves = generate_ordered_legal_moves(board)

    for move in legal_moves:
        # Measure static evaluation of the hypothetical immediate position
        board.push(move)
        static_leaf_score = evaluate_position(board, heuristic_mode)

        if algorithm == "minimax":
            score = pure_minimax(board, depth - 1, not is_white, stats, heuristic_mode)
        else:
            score = alphabeta_search(board, depth - 1, alpha, beta, not is_white, stats, heuristic_mode)
        board.pop()

        # Generate standard algebraic notation (SAN) safely
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()

        candidate_records.append((move, san, score, static_leaf_score))

        if is_white:
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)

    elapsed_sec = time.perf_counter() - start_time
    stats.elapsed_ms = elapsed_sec * 1000.0
    stats.best_move = best_move
    stats.best_score = best_score

    # Sort candidate moves best-first according to the player's goal
    candidate_records.sort(key=lambda item: item[2], reverse=is_white)
    stats.top_moves = [
        {
            "uci": m.uci(),
            "san": s,
            "minimax_score": sc,
            "static_score": st,
        }
        for m, s, sc, st in candidate_records[:6]
    ]

    # Generate academic breakdown and explanation of resulting board position
    if best_move:
        board.push(best_move)
        breakdown = evaluate_position_breakdown(board, heuristic_mode)
        board.pop()
        stats.breakdown = breakdown
        stats.explanation = format_viva_explanation(best_move, best_score, depth, algorithm, heuristic_mode, breakdown, is_white)
    else:
        stats.explanation = "No legal moves available."

    return stats


def format_viva_explanation(move, score, depth, algorithm, mode, breakdown, is_white):
    """Formats an academically rigorous explanation for viva presentation."""
    side = "White" if is_white else "Black"
    alg_name = "Minimax with Alpha-Beta Pruning" if algorithm == "alphabeta" else "Pure Depth-First Minimax"

    if abs(score) >= CHECKMATE_SCORE:
        steps_to_mate = (CHECKMATE_SCORE + depth - abs(score)) if abs(score) > CHECKMATE_SCORE else 1
        winner = "White" if score > 0 else "Black"
        return f"{side} selected {move.uci()} utilizing {alg_name} (depth {depth}). A forced checkmate for {winner} was identified within {steps_to_mate} plies."

    adv = "White" if score > 0 else "Black" if score < 0 else "Neither"
    pawn_equiv = abs(score) / 100.0

    lines = [
        f"Algorithm: {alg_name} explored {depth} plies ahead.",
        f"Selected move {move.uci()} achieves backed-up Minimax value of {score:+d} cp (~{pawn_equiv:.2f} pawns favor {adv}).",
    ]

    if mode == "full":
        lines.append(
            f"Evaluation components: Material ({breakdown['material']:+d} cp), "
            f"Piece Placement ({breakdown['pst']:+d} cp), "
            f"Center Pressure ({breakdown['center']:+d} cp), "
            f"Mobility ({breakdown['mobility']:+d} cp), "
            f"King Safety ({breakdown['king_safety']:+d} cp)."
        )
    else:
        lines.append(f"Evaluation mode: Material-only baseline ({breakdown['material']:+d} cp).")

    return " ".join(lines)


# ---------------------------------------------------------------------------
#  Experiment Mode: Multi-Depth & Heuristic Comparison
# ---------------------------------------------------------------------------

def run_experiment(board, depths=(1, 2, 3), algorithm="alphabeta", heuristic_mode="full"):
    """
    Evaluates the exact same position across multiple depths.
    Demonstrates search time scaling, node expansion rate, and decision consistency.
    """
    results = []
    for d in depths:
        b_copy = board.copy()
        stats = find_best_move(b_copy, depth=d, algorithm=algorithm, heuristic_mode=heuristic_mode)
        results.append(stats)
    return results


def compare_heuristics_benchmark(board, depth=2):
    """
    Runs a side-by-side comparison between Material-Only baseline and Full Heuristic.
    """
    baseline_stats = find_best_move(board.copy(), depth=depth, algorithm="alphabeta", heuristic_mode="material")
    full_stats     = find_best_move(board.copy(), depth=depth, algorithm="alphabeta", heuristic_mode="full")
    return {
        "baseline": baseline_stats,
        "full": full_stats,
    }


# ---------------------------------------------------------------------------
#  Teaching & Demonstration: Game Tree Snapshot Generation
# ---------------------------------------------------------------------------

def build_tree_snapshot(board, depth=2, max_children=4, heuristic_mode="full"):
    """
    Constructs a JSON-serializable game tree representation limited in depth
    and branching factor so students and examiners can inspect DFS recursion,
    leaf static evaluations, and backed-up Minimax propagation.
    """
    target_depth = max(1, min(depth, 3))
    root_maximizing = (board.turn == chess.WHITE)

    def _recurse(b, current_ply, is_max):
        static_score = evaluate_position(b, heuristic_mode)
        node = {
            "fen": b.fen(),
            "ply": current_ply,
            "static_score": static_score,
            "is_max": is_max,
            "move": None,
            "san": None,
            "children": [],
            "minimax_score": None,
        }

        if current_ply >= target_depth or b.is_game_over():
            node["minimax_score"] = static_score
            return node

        moves = generate_ordered_legal_moves(b)[:max_children]
        child_scores = []
        for m in moves:
            san = b.san(m)
            b.push(m)
            child = _recurse(b, current_ply + 1, not is_max)
            child["move"] = m.uci()
            child["san"] = san
            node["children"].append(child)
            child_scores.append(child["minimax_score"])
            b.pop()

        if child_scores:
            node["minimax_score"] = max(child_scores) if is_max else min(child_scores)
        else:
            node["minimax_score"] = static_score

        return node

    return _recurse(board.copy(), 0, root_maximizing)


# ---------------------------------------------------------------------------
#  Command-Line Interface (for testing and direct demonstration)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic Chess Heuristic Search AI")
    parser.add_argument("--fen", type=str, default=chess.STARTING_FEN, help="FEN string of position")
    parser.add_argument("--depth", type=int, default=3, help="Search depth in plies")
    parser.add_argument("--alg", choices=["alphabeta", "minimax"], default="alphabeta", help="Algorithm choice")
    parser.add_argument("--heuristic", choices=["full", "material"], default="full", help="Heuristic mode")
    parser.add_argument("--experiment", action="store_true", help="Run multi-depth experiment")

    args = parser.parse_args()
    demo_board = chess.Board(args.fen)

    print("=" * 65)
    print("  AI HEURISTIC SEARCH CHESS DEMO")
    print("=" * 65)
    print(f"Position FEN: {args.fen}")
    print(f"Side to move: {'White' if demo_board.turn == chess.WHITE else 'Black'}")

    if args.experiment:
        print("\nRunning multi-depth experiment across depths [1, 2, 3]...")
        exp_results = run_experiment(demo_board, depths=[1, 2, 3], algorithm=args.alg, heuristic_mode=args.heuristic)
        print("\nDepth | Best Move | Minimax Score | Nodes Visited | Leaves | Elapsed (ms)")
        print("-" * 65)
        for r in exp_results:
            move_str = r.best_move.uci() if r.best_move else "None"
            print(f"  {r.depth}   |   {move_str:7} |    {r.best_score:+7d}    |     {r.nodes_visited:7d}   |  {r.leaves_evaluated:5d} | {r.elapsed_ms:8.2f} ms")
    else:
        print(f"\nSearching to depth {args.depth} using {args.alg.upper()} ({args.heuristic} heuristic)...")
        stats = find_best_move(demo_board, depth=args.depth, algorithm=args.alg, heuristic_mode=args.heuristic)
        print(f"Chosen move: {stats.best_move}")
        print(f"Minimax score: {stats.best_score:+d} cp")
        print(f"Nodes visited: {stats.nodes_visited}")
        print(f"Leaves evaluated: {stats.leaves_evaluated}")
        if args.alg == "alphabeta":
            print(f"Alpha-Beta cuts: {stats.alpha_beta_cuts}")
        print(f"Elapsed time: {stats.elapsed_ms:.2f} ms")
        print(f"\nExplanation:\n{stats.explanation}")
        print("\nTop Candidate Moves:")
        for cand in stats.top_moves:
            print(f"  Move {cand['san']} ({cand['uci']}): Minimax Score = {cand['minimax_score']:+d} cp | Static Leaf Score = {cand['static_score']:+d} cp")
