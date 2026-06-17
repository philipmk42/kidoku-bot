"""
Inference module for Kidoku Bot

Provides get_bot_move() function that loads KidokuPolicyNet
and returns a human-like move using temperature sampling.
"""

import torch
import torch.nn.functional as F
import chess
import random
from pathlib import Path

from src.model import KidokuPolicyNet
from src.data.process_pgn import board_to_tensor, move_to_index, index_to_move, MOVE_SPACE_SIZE


class KidokuBot:
    def __init__(self, model_path: str = None, device: str = "cpu", temperature: float = 1.8):
        """
        Args:
            model_path: Path to saved .pth file. If None, uses random weights (for testing).
            device: "cpu" or "cuda"
            temperature: Higher = more random/human-like, Lower = more deterministic
        """
        self.device = device
        self.temperature = temperature

        self.model = KidokuPolicyNet(num_moves=MOVE_SPACE_SIZE).to(device)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print("Using random weights (model not trained yet)")

        self.model.eval()

    def get_move(self, board: chess.Board) -> str:
        """
        Returns a UCI move string for the given board position.
        Uses temperature sampling over legal moves for human-like play.
        """
        # 1. Mate in 1 check (humans usually take free mates)
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        # 2. Get model prediction
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)[0]   # shape: (4672,)

        # 3. Filter only legal moves
        legal_moves = list(board.legal_moves)
        legal_indices = []
        legal_logits = []

        for move in legal_moves:
            idx = move_to_index(move)
            if idx < MOVE_SPACE_SIZE:
                legal_indices.append(idx)
                legal_logits.append(logits[idx])

        if not legal_logits:
            # Fallback: pick random legal move
            return random.choice([m.uci() for m in legal_moves])

        legal_logits = torch.stack(legal_logits)

        # 4. Temperature sampling
        probs = F.softmax(legal_logits / self.temperature, dim=0)

        # Sample one move
        sampled = torch.multinomial(probs, num_samples=1).item()
        chosen_index = legal_indices[sampled]

        # Convert back to UCI
        chosen_move = index_to_move(chosen_index, board)
        if chosen_move is None:
            chosen_move = random.choice(legal_moves)

        return chosen_move.uci()


# Quick test function
def test_bot():
    bot = KidokuBot(temperature=1.8)
    board = chess.Board()
    
    print("Testing KidokuBot on starting position...")
    for _ in range(5):
        move = bot.get_move(board)
        print(f"Move: {move}")
        board.push_uci(move)
        if board.is_game_over():
            break
    
    print("\nGame ended.")
    print(board)


if __name__ == "__main__":
    test_bot()
