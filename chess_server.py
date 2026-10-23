"""
chess_server.py  -  Local HTTP Server & JSON API for Chess AI
=============================================================
Academic Project: Provides a web interface and RESTful JSON API to:
- Play Human vs AI (as White or Black), Human vs Human, or AI vs AI.
- Configure search depth (Easy=1-2, Medium=3, Hard=4).
- Select between Pure Minimax (DFS) and Minimax with Alpha-Beta Pruning.
- Select between Baseline Material-Only evaluation and Enhanced Heuristic.
- Inspect real-time search diagnostics: nodes visited, leaves evaluated,
  pruning cutoffs, execution time (ms), static vs backed-up scores.
- Explore interactive game tree visualizations.
- Run multi-depth benchmarks and comparative experiments on any position.

Run:
    python chess_server.py [--port 8765]
"""

import json
import os
import sys
import threading
import urllib.parse
import http.server
import webbrowser

import chess
import chess.pgn

import chess_ai as ai

# ---------------------------------------------------------------------------
#  Global Game State (Thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_board = chess.Board()
_last_stats = None
_move_history = []  # [{"uci": "e2e4", "san": "e4", "turn": "white"}, ...]

# Academic preset test positions
PRESETS = [
    {
        "id": "start",
        "name": "Standard Starting Position",
        "fen": chess.STARTING_FEN,
        "description": "Standard chess initial layout (all pieces on home ranks)."
    },
    {
        "id": "mate_in_1_white",
        "name": "Scholar's Mate Threat (White to Mate in 1)",
        "fen": "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4",
        "description": "White Queen at f3 and Bishop at c4 target f7 for immediate checkmate (Qxf7#)."
    },
    {
        "id": "mate_in_1_black",
        "name": "Fool's Mate Defense (Black to Mate in 1)",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "description": "White weakened the e1-h4 diagonal (f3, g4); Black Queen can deliver Qh4#."
    },
    {
        "id": "fork_tactic",
        "name": "Royal Knight Fork Tactic",
        "fen": "r1b1k2r/pppp1ppp/8/4q3/1b6/2N5/PPP1BPPP/R1BQK2R w KQkq - 0 1",
        "description": "Tactical middlegame position testing positional piece evaluation and mobility."
    },
    {
        "id": "pawn_promotion",
        "name": "Pawn Endgame & Promotion",
        "fen": "8/4P3/8/8/8/6k1/8/6K1 w - - 0 1",
        "description": "White pawn on 7th rank (e7) promotes to Queen on e8."
    },
    {
        "id": "stalemate_trap",
        "name": "Stalemate Defense Test",
        "fen": "k7/8/1K6/8/8/8/8/1Q6 w - - 0 1",
        "description": "White must avoid accidental stalemate (e.g. Qb5 stalemate vs Qb7# mate)."
    }
]


def _board_to_json(board):
    """Serialize board state into a clean dictionary for browser consumption."""
    squares = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            squares[chess.square_name(sq)] = {
                "type": piece.piece_type,
                "color": "white" if piece.color == chess.WHITE else "black",
                "symbol": piece.symbol(),
            }

    legal = []
    for m in board.legal_moves:
        legal.append({
            "from": chess.square_name(m.from_square),
            "to": chess.square_name(m.to_square),
            "uci": m.uci(),
            "san": board.san(m),
            "promotion": chess.piece_name(m.promotion) if m.promotion else None,
        })

    return {
        "squares": squares,
        "legal_moves": legal,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_insufficient": board.is_insufficient_material(),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "fullmove": board.fullmove_number,
        "fen": board.fen(),
        "move_history": list(_move_history),
        "can_undo": len(board.move_stack) > 0,
    }


def _stats_to_json(stats):
    if stats is None:
        return None
    return {
        "depth": stats.depth,
        "nodes_visited": stats.nodes_visited,
        "leaves_evaluated": stats.leaves_evaluated,
        "alpha_beta_cuts": stats.alpha_beta_cuts,
        "elapsed_ms": round(stats.elapsed_ms, 2),
        "best_move": stats.best_move.uci() if stats.best_move else None,
        "best_san": stats.top_moves[0]["san"] if (stats.top_moves and "san" in stats.top_moves[0]) else (stats.best_move.uci() if stats.best_move else None),
        "best_score": stats.best_score,
        "top_moves": stats.top_moves,
        "algorithm": stats.algorithm,
        "heuristic_mode": stats.heuristic_mode,
        "explanation": stats.explanation,
        "breakdown": stats.breakdown,
    }


# ---------------------------------------------------------------------------
#  HTTP Request Handler
# ---------------------------------------------------------------------------

class ChessHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # Suppress default request flooding in stdout

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(404, "File not found")

    def _parse_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── GET Handlers ────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
            self._send_file(index_path, "text/html; charset=utf-8")

        elif path.startswith("/img/"):
            rel_path = path.lstrip("/")
            fpath = os.path.join(os.path.dirname(__file__), rel_path)
            ctype = "image/svg+xml" if fpath.endswith(".svg") else "image/png"
            if os.path.isfile(fpath):
                self._send_file(fpath, ctype)
            else:
                self.send_error(404, "Image not found")

        elif path == "/api/state":
            with _lock:
                self._send_json({
                    "board": _board_to_json(_board),
                    "last_stats": _stats_to_json(_last_stats),
                })

        elif path == "/api/presets":
            self._send_json(PRESETS)

        elif path == "/api/tree":
            depth = int(params.get("depth", ["2"])[0])
            mode = params.get("heuristic", ["full"])[0]
            with _lock:
                b = _board.copy()
            snapshot = ai.build_tree_snapshot(b, depth=depth, max_children=4, heuristic_mode=mode)
            self._send_json(snapshot)

        else:
            self.send_error(404, "Not Found")

    # ── POST Handlers ───────────────────────────────────────────────────────

    def do_POST(self):
        global _board, _last_stats, _move_history
        path = urllib.parse.urlparse(self.path).path
        body = self._parse_body()

        # ── Human Move ──────────────────────────────────────────────────────
        if path == "/api/move":
            uci = body.get("uci", "")
            with _lock:
                try:
                    move = chess.Move.from_uci(uci)
                    if move not in _board.legal_moves:
                        self._send_json({"error": "Illegal move attempted."}, 400)
                        return

                    san = _board.san(move)
                    turn_name = "white" if _board.turn == chess.WHITE else "black"
                    _board.push(move)
                    _move_history.append({"uci": uci, "san": san, "turn": turn_name})
                    _last_stats = None

                    resp = {
                        "board": _board_to_json(_board),
                        "last_stats": None,
                    }
                except Exception as e:
                    self._send_json({"error": str(e)}, 400)
                    return
            self._send_json(resp)

        # ── AI Move ─────────────────────────────────────────────────────────
        elif path == "/api/ai_move":
            depth = int(body.get("depth", 3))
            algorithm = body.get("algorithm", "alphabeta")
            heuristic_mode = body.get("heuristic_mode", "full")

            with _lock:
                if _board.is_game_over():
                    self._send_json({"error": "Game is already over."}, 400)
                    return
                b_clone = _board.copy()

            # Execute search
            stats = ai.find_best_move(b_clone, depth=depth, algorithm=algorithm, heuristic_mode=heuristic_mode)

            with _lock:
                if stats.best_move and stats.best_move in _board.legal_moves:
                    san = _board.san(stats.best_move)
                    turn_name = "white" if _board.turn == chess.WHITE else "black"
                    _board.push(stats.best_move)
                    _move_history.append({"uci": stats.best_move.uci(), "san": san, "turn": turn_name})
                    _last_stats = stats
                    resp = {
                        "board": _board_to_json(_board),
                        "last_stats": _stats_to_json(stats),
                    }
                else:
                    resp = {"error": "AI could not find a legal move."}
            self._send_json(resp)

        # ── Undo Move ────────────────────────────────────────────────────────
        elif path == "/api/undo":
            count = int(body.get("count", 1))
            with _lock:
                for _ in range(count):
                    if len(_board.move_stack) > 0:
                        _board.pop()
                        if _move_history:
                            _move_history.pop()
                _last_stats = None
                resp = {
                    "board": _board_to_json(_board),
                    "last_stats": None,
                }
            self._send_json(resp)

        # ── New Game / Load FEN ──────────────────────────────────────────────
        elif path == "/api/new_game":
            fen = body.get("fen", chess.STARTING_FEN)
            with _lock:
                try:
                    _board = chess.Board(fen)
                except Exception:
                    _board = chess.Board()
                _last_stats = None
                _move_history = []
                resp = {
                    "board": _board_to_json(_board),
                    "last_stats": None,
                }
            self._send_json(resp)

        # ── Multi-Depth Experiment Benchmark ─────────────────────────────────
        elif path == "/api/experiment":
            depths = body.get("depths", [1, 2, 3])
            algorithm = body.get("algorithm", "alphabeta")
            heuristic_mode = body.get("heuristic_mode", "full")

            with _lock:
                b_copy = _board.copy()

            results = ai.run_experiment(b_copy, depths=depths, algorithm=algorithm, heuristic_mode=heuristic_mode)
            self._send_json([_stats_to_json(s) for s in results])

        # ── Heuristic Comparison Benchmark ───────────────────────────────────
        elif path == "/api/compare_heuristics":
            depth = int(body.get("depth", 2))
            with _lock:
                b_copy = _board.copy()
            comparison = ai.compare_heuristics_benchmark(b_copy, depth=depth)
            self._send_json({
                "baseline": _stats_to_json(comparison["baseline"]),
                "full": _stats_to_json(comparison["full"]),
            })

        else:
            self.send_error(404, "Endpoint not found")


# ---------------------------------------------------------------------------
#  Server Launch Routine
# ---------------------------------------------------------------------------

def run_server(port=8765, open_browser=True):
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        if arg == "--no-browser":
            open_browser = False

    server = http.server.HTTPServer(("", port), ChessHandler)
    url = f"http://localhost:{port}"

    print("=" * 65)
    print("  CHESS ARTIFICIAL INTELLIGENCE SERVER")
    print("=" * 65)
    print(f"  Server URL:    {url}")
    print(f"  Port:          {port}")
    print("  Serving GUI:   static/index.html")
    print("  Serving SVGs:  img/white and img/black")
    print("  Press Ctrl+C to terminate.")
    print("=" * 65)

    if open_browser:
        def _launch():
            import time
            time.sleep(0.8)
            webbrowser.open(url)
        threading.Thread(target=_launch, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Chess AI server gracefully stopped.")


if __name__ == "__main__":
    run_server()
