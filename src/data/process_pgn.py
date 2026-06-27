"""
Data Processing Pipeline for Kidoku Bot

Converts PGN games into training examples for behavioral cloning.

We use a FIXED policy head representation:
- Every possible move is represented as: from_square * 64 + to_square
- Promotions add an offset (we support queen, rook, bishop, knight underpromotion)

This allows the model to output ANY legal move, not just moves seen in training.

CHANGES vs the original:
- Every example is tagged with a global game_id so training can split BY GAME
  (no positions from the same game leaking across train/val).
- Defaults now process only the target player's side (kidoku = Black) over the
  full merged PGN, instead of both colors on a 50-game sample.
"""

import chess
import chess.pgn
import numpy as np
from pathlib import Path
from typing import List, Tuple
import torch
from tqdm import tqdm


# ==================== BOARD REPRESENTATION ====================

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to a 14-channel tensor.
    Channels:
        0-5:   White pieces (P, N, B, R, Q, K)
        6-11:  Black pieces
        12:    Turn (1 if white to move)
        13:    Castling rights
    """
    tensor = np.zeros((14, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = square // 8
        col = square % 8
        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6
        tensor[channel, row, col] = 1.0

    # Turn indicator
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Castling rights (simplified)
    if board.castling_rights > 0:
        tensor[13, :, :] = 1.0

    return tensor


# ==================== FIXED MOVE REPRESENTATION ====================

MOVE_SPACE_SIZE = 4672

def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess.Move into a fixed integer index (0 to 4671).

    Encoding scheme (simple but effective):
    - Normal moves:     from_square * 64 + to_square   -> range 0-4095
    - Underpromotions:  4096 + (promo_piece * 64) + to_square
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_offset = (move.promotion - 2) * 64 + to_sq   # 2=Queen, 3=Rook...
        return 4096 + promo_offset
    else:
        return from_sq * 64 + to_sq


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Reverse of move_to_index. Used during inference."""
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq)
    else:
        promo_offset = index - 4096
        promo_type = (promo_offset // 8) + 2  # 2=Queen, 3=Rook, 4=Bishop, 5=Knight
        file = promo_offset % 8
        for move in board.legal_moves:
            if move.promotion == promo_type and (move.to_square % 8) == file:
                return move
        return None


# ==================== DATA PROCESSING ====================

def process_game(game: chess.pgn.Game, player_color: chess.Color = None) -> List[Tuple[np.ndarray, int]]:
    """
    Process a single game and return list of (board_tensor, move_index) pairs.

    If player_color is None, take moves from both sides.
    If player_color is chess.WHITE or chess.BLACK, take ONLY that side's moves
    (the positions where it is that player's turn and the move they actually made).
    """
    examples = []
    board = game.board()

    for move in game.mainline_moves():
        if player_color is None or board.turn == player_color:
            tensor = board_to_tensor(board)
            move_idx = move_to_index(move)
            examples.append((tensor, move_idx))
        board.push(move)

    return examples


def process_pgn_file(
    pgn_path: str,
    output_dir: str,
    player_color: chess.Color = chess.BLACK,
    max_games: int = None,
    save_every: int = 200
) -> None:
    """
    Process an entire PGN file and save training examples as .pt chunks.

    Each saved chunk now contains:
        boards   : (N, 14, 8, 8) float tensor
        moves    : (N,)          long tensor (move index)
        game_ids : (N,)          long tensor (which game each example came from)

    game_ids are global and monotonic across the whole file, so the trainer can
    split by game without any leakage.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []   # list of (tensor, move_idx, game_id)
    game_count = 0

    print(f"Processing PGN: {pgn_path}")
    side = {None: "both colors", chess.WHITE: "White", chess.BLACK: "Black"}[player_color]
    print(f"Collecting moves for: {side}")

    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        pbar = tqdm(desc="Games processed")

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            for tensor, move_idx in process_game(game, player_color=player_color):
                all_examples.append((tensor, move_idx, game_count))
            game_count += 1
            pbar.update(1)

            if max_games and game_count >= max_games:
                break

            if len(all_examples) >= save_every * 40:  # ~40 plies per game avg
                save_chunk(all_examples, output_path, game_count)
                all_examples = []

        pbar.close()

    if all_examples:
        save_chunk(all_examples, output_path, game_count)

    print(f"\nFinished processing {game_count} games.")


def save_chunk(examples: List[Tuple[np.ndarray, int]], output_dir: Path, game_count: int):
    """Save a chunk of (board, move, game_id) examples as a .pt file."""
    if not examples:
        return

    boards, moves, game_ids = zip(*examples)
    boards_tensor = torch.from_numpy(np.array(boards))
    moves_tensor = torch.tensor(moves, dtype=torch.long)
    game_ids_tensor = torch.tensor(game_ids, dtype=torch.long)

    filename = output_dir / f"chunk_{game_count:05d}.pt"
    torch.save({
        "boards": boards_tensor,
        "moves": moves_tensor,
        "game_ids": game_ids_tensor,
    }, filename)

    print(f"Saved {len(examples)} examples -> {filename.name}")


if __name__ == "__main__":
    # Black-only (kidoku's side), full merged set.
    # IMPORTANT: delete old chunks first -- they have no game_ids and the
    # trainer will refuse to load them:
    #   Remove-Item .\data\processed\*.pt
    process_pgn_file(
        pgn_path="data/kidoku_all.pgn",
        output_dir="data/processed",
        player_color=chess.BLACK,
        max_games=None,
        save_every=200,
    )
