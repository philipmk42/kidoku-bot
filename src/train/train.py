"""
Training script for KidokuPolicyNet

Usage:
    python -m src.train.train

CHANGES vs the original:
- Splits BY GAME (whole games held out for validation), so val move-match is a
  real generalization number, not memorized positions.
- Reports validation move-match (top-1 and top-3) and the train-val gap each epoch.
- Saves models/kidoku_policy_best.pth on best VAL score (not the memorized final
  epoch). Use that checkpoint for play/inference.
- Adds weight decay and EARLY STOPPING: training halts once val move-match has not
  improved for `patience` epochs, so you stop at the plateau instead of memorizing.
"""

import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from src.model import KidokuPolicyNet


class ChessDataset(Dataset):
    """Loads preprocessed board tensors + move targets + game ids."""
    def __init__(self, processed_dir: str):
        self.files = sorted(glob.glob(str(Path(processed_dir) / "chunk_*.pt")))
        if not self.files:
            raise FileNotFoundError(
                f"No chunk_*.pt files in {processed_dir}. Run process_pgn.py first."
            )

        boards, moves, game_ids = [], [], []
        print(f"Loading {len(self.files)} data chunks...")
        for f in self.files:
            data = torch.load(f, map_location="cpu")
            if "game_ids" not in data:
                raise KeyError(
                    f"{f} has no 'game_ids' (old pipeline output). "
                    "Delete data/processed/*.pt and re-run process_pgn.py."
                )
            boards.append(data["boards"])
            moves.append(data["moves"])
            game_ids.append(data["game_ids"])

        self.boards = torch.cat(boards, dim=0)
        self.moves = torch.cat(moves, dim=0)
        self.game_ids = torch.cat(game_ids, dim=0)
        n_games = int(self.game_ids.max().item()) + 1
        print(f"Loaded {len(self.boards)} examples from {n_games} games.")

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.moves[idx]


def split_by_game(game_ids: torch.Tensor, val_frac: float = 0.15, seed: int = 42):
    """
    Return (train_idx, val_idx, n_train_games, n_val_games).

    Holds out whole GAMES for validation so no position from a validation game
    ever appears in training. This is what makes the val number trustworthy.
    """
    unique = torch.unique(game_ids)
    g = torch.Generator().manual_seed(seed)
    perm = unique[torch.randperm(len(unique), generator=g)]
    n_val = max(1, int(len(unique) * val_frac))
    val_games = set(perm[:n_val].tolist())

    is_val = torch.tensor([gid.item() in val_games for gid in game_ids])
    val_idx = torch.nonzero(is_val, as_tuple=True)[0].tolist()
    train_idx = torch.nonzero(~is_val, as_tuple=True)[0].tolist()
    return train_idx, val_idx, len(unique) - n_val, n_val


@torch.no_grad()
def evaluate(model, loader, device):
    """Validation move-match: how often the model picks kidoku's actual move."""
    model.eval()
    top1 = top3 = total = 0
    for boards, moves in loader:
        boards = boards.to(device)
        moves = moves.to(device)
        logits = model(boards)

        top1 += (logits.argmax(dim=1) == moves).sum().item()
        top3_idx = logits.topk(3, dim=1).indices
        top3 += (top3_idx == moves.unsqueeze(1)).any(dim=1).sum().item()
        total += moves.size(0)

    return 100 * top1 / total, 100 * top3 / total


def train(
    data_dir: str = "data/processed",
    batch_size: int = 128,
    epochs: int = 40,            # max epochs; early stopping usually ends sooner
    lr: float = 1e-3,
    weight_decay: float = 2e-2,  # L2 regularization; raise to fight overfitting, lower if underfitting
    patience: int = 6,           # stop if val move-match doesn't improve for this many epochs
    val_frac: float = 0.15,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    print(f"Using device: {device}")

    # Data + by-game split
    full = ChessDataset(data_dir)
    train_idx, val_idx, n_train_games, n_val_games = split_by_game(
        full.game_ids, val_frac=val_frac, seed=seed
    )
    train_set = Subset(full, train_idx)
    val_set = Subset(full, val_idx)
    print(
        f"Split by game -> train: {len(train_idx)} examples / {n_train_games} games | "
        f"val: {len(val_idx)} examples / {n_val_games} games"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = KidokuPolicyNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Path("models").mkdir(exist_ok=True)
    best_val = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for boards, moves in pbar:
            boards = boards.to(device)
            moves = moves.to(device)

            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, moves)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == moves).sum().item()
            total += moves.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "train_acc": f"{100 * correct / total:.2f}%",
            })

        train_acc = 100 * correct / total
        val_top1, val_top3 = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | "
            f"Train: {train_acc:.2f}% | "
            f"Val move-match: {val_top1:.2f}% (top-3 {val_top3:.2f}%) | "
            f"gap {train_acc - val_top1:+.1f}"
        )

        # Keep the checkpoint that generalizes best, not the most memorized one.
        if val_top1 > best_val:
            best_val = val_top1
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save(model.state_dict(), "models/kidoku_policy_best.pth")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"\nEarly stop: no val improvement for {patience} epochs.")
                break

    torch.save(model.state_dict(), "models/kidoku_policy_final.pth")
    print(f"\nDone. Best val move-match: {best_val:.2f}% (epoch {best_epoch}).")
    print("Use models/kidoku_policy_best.pth for play/inference (NOT the final epoch).")


if __name__ == "__main__":
    train()
