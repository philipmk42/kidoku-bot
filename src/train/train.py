"""
Training script for KidokuPolicyNet

Usage:
    python -m src.train.train
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import glob

from src.model import KidokuPolicyNet


class ChessDataset(Dataset):
    """Simple dataset that loads preprocessed board tensors + move targets."""
    def __init__(self, processed_dir: str):
        self.files = sorted(glob.glob(str(Path(processed_dir) / "chunk_*.pt")))
        self.boards = []
        self.moves = []

        print(f"Loading {len(self.files)} data chunks...")
        for f in self.files:
            data = torch.load(f, map_location="cpu")
            self.boards.append(data["boards"])
            self.moves.append(data["moves"])

        self.boards = torch.cat(self.boards, dim=0)
        self.moves = torch.cat(self.moves, dim=0)
        print(f"Loaded {len(self.boards)} training examples.")

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.moves[idx]


def train(
    data_dir: str = "data/processed",
    batch_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")

    # Data
    dataset = ChessDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    model = KidokuPolicyNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for boards, moves in pbar:
            boards = boards.to(device)
            moves = moves.to(device)

            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, moves)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == moves).sum().item()
            total += moves.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Top-1 Accuracy: {accuracy:.2f}%")

        # Save checkpoint
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"models/kidoku_policy_epoch{epoch+1}.pth")

    print("\nTraining finished!")
    torch.save(model.state_dict(), "models/kidoku_policy_final.pth")
    print("Final model saved to models/kidoku_policy_final.pth")


if __name__ == "__main__":
    train()
