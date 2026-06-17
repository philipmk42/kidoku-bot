"""
Data Processing Pipeline for Kidoku Bot

Converts PGN games into training examples for behavioral cloning.

We use a FIXED policy head representation:
- Every possible move is represented as: from_square * 64 + to_square
- Promotions add an offset (we support queen, rook, bishop, knight underpromotion)

This allows the model to output ANY legal move, not just moves seen in training.
"""

import chess
import chess.pgn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterator
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

# Total possible moves in our encoding:
# Normal moves: 64 * 64 = 4096
# Promotions: 4 promotion types * 16 possible promotion moves (from 7th/2nd rank)
# We'll use a simple 4672-size space (common in chess nets)

MOVE_SPACE_SIZE = 4672

def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess.Move into a fixed integer index (0 to 4671).

    Encoding scheme (simple but effective):
    - Normal moves:     from_square * 64 + to_square   → range 0–4095
    - Underpromotions:  4096 + (promo_piece * 64) + to_square
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Only encode underpromotions specially (Queen promotion is most common)
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
        # Promotion
        promo_offset = index - 4096
        promo_type = (promo_offset // 8) + 2  # 2=Queen, 3=Rook, 4=Bishop, 5=Knight
        file = promo_offset % 8

        # Find a legal promotion move on that file (simplified)
        for move in board.legal_moves:
            if move.promotion == promo_type and (move.to_square % 8) == file:
                return move
        return None


# ==================== DATA PROCESSING ====================

def process_game(game: chess.pgn.Game, player_color: chess.Color = None) -> List[Tuple[np.ndarray, int]]:
    """
    Process a single game and return list of (board_tensor, move_index) pairs.

    If player_color is None, we take moves from both sides.
    If player_color is specified (chess.WHITE or chess.BLACK), we only take that side's moves.
    """
    examples = []
    board = game.board()

    for move in game.mainline_moves():
        # Only collect positions where it's the target player's turn
        if player_color is None or board.turn == player_color:
            tensor = board_to_tensor(board)
            move_idx = move_to_index(move)
            examples.append((tensor, move_idx))

        board.push(move)

    return examples


def process_pgn_file(
    pgn_path: str,
    output_dir: str,
    player_color: chess.Color = None,
    max_games: int = None,
    save_every: int = 500
) -> None:
    """
    Process an entire PGN file and save training examples as .pt files.

    Args:
        pgn_path: Path to the PGN file
        output_dir: Where to save processed tensors
        player_color: If set, only process moves by this color
        max_games: Limit number of games (for testing)
        save_every: Save intermediate files every N games
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []
    game_count = 0

    print(f"Processing PGN: {pgn_path}")

    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        pbar = tqdm(desc="Games processed")

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            examples = process_game(game, player_color=player_color)
            all_examples.extend(examples)
            game_count += 1
            pbar.update(1)

            if max_games and game_count >= max_games:
                break

            # Periodically save to avoid memory issues
            if len(all_examples) >= save_every * 40:  # ~40 moves per game avg
                save_chunk(all_examples, output_path, game_count)
                all_examples = []

        pbar.close()

    # Save remaining examples
    if all_examples:
        save_chunk(all_examples, output_path, game_count)

    print(f"\nFinished processing {game_count} games.")
    print(f"Total training examples: {game_count * 35} (approximate)")


def save_chunk(examples: List[Tuple[np.ndarray, int]], output_dir: Path, game_count: int):
    """Save a chunk of examples as a .pt file."""
    if not examples:
        return

    boards, moves = zip(*examples)
    boards_tensor = torch.from_numpy(np.array(boards))
    moves_tensor = torch.tensor(moves, dtype=torch.long)

    filename = output_dir / f"chunk_{game_count:05d}.pt"
    torch.save({
        "boards": boards_tensor,
        "moves": moves_tensor
    }, filename)

    print(f"Saved {len(examples)} examples → {filename.name}")


if __name__ == "__main__":
    # Example usage
    process_pgn_file(
        pgn_path="data/raw/kidoku_games.pgn",
        output_dir="data/processed",
        player_color=None,        # Process both colors for now
        max_games=50,             # Start small for testing
        save_every=200
    )
