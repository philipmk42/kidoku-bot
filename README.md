# Kidoku Bot

A neural chess bot that learns to **play like a specific human** rather than to play well.
It clones the move choices of `kidokuismyname`, a roughly 670-rated Chess.com player,
using behavioral cloning on about 1,100 of their games.

The goal is not strength. It is faithfulness: given a position, the bot tries to play the
move *that player* would actually play, habits and mistakes included.

## Results

Measured on held-out games (see "How it is evaluated" below):

| Metric | Score |
| --- | --- |
| Top-1 move-match | **52.6%** |
| Top-3 move-match | **63.1%** |

On positions from games it has never seen, the model picks kidoku's exact move about 53% of
the time, and kidoku's move is among its top 3 about 63% of the time. Trained on ~1,100
games (Black only), ~30k training positions, ResNet-style policy network.

## How it is evaluated (and why the number is honest)

Move-match is measured with a **by-game** train/validation split: 15% of whole games are
held out, so no position from a validation game ever appears in training.

This matters. A single chess game contains ~30 to 40 highly correlated positions. Splitting
*individual positions* into train and validation leaks near-duplicate positions across the
two sets and inflates the score. Splitting by *game* is what makes 53% a real generalization
number instead of memorization. The split is seeded, so the result is reproducible.

## Honest limitations

- **Black only.** The bot is trained on kidoku's games as Black, so it plays Black. The UI
  assigns the human player White.
- **Residual overfitting.** Train accuracy reaches ~80% while validation plateaus near 53%,
  so a train/val gap remains. Validation flattens with more epochs, which suggests the model
  is near the ceiling for this amount of data; more games is the most likely way to push it
  further.
- **Imitation ceiling.** A ~670-rated player is inconsistent and will play different moves in
  similar positions, so exact top-1 match has a natural ceiling well below 100%. Top-3 is
  arguably the fairer measure of how well the style is captured.
- This is not a strong engine. By design it reproduces the target player's mistakes.

## Architecture

- **Input:** 14 x 8 x 8 board tensor (piece planes, side to move, castling rights)
- **Backbone:** 6 residual blocks, 64 channels
- **Policy head:** logits over a fixed 4,672-move encoding (from-square x to-square plus
  underpromotions), with dropout regularization
- ~3.9M parameters, PyTorch

## Project structure

```
src/
  data/process_pgn.py    PGN -> training tensors. Collects only the target player's moves
                         and tags each example with a game id so training can split by game.
  model/policy_net.py    KidokuPolicyNet (ResNet-style policy network).
  train/train.py         Training: by-game split, validation move-match (top-1 / top-3),
                         early stopping, best-validation checkpointing.
  ui/                    Pygame interface to play against the bot.
train.py                 Entry point (calls src.train.train).
data/kidoku_all.pgn      Merged games of the target player (not committed).
models/                  Saved checkpoints (not committed).
```

## Usage

```bash
# 1. Collect the target player's games into data/kidoku_all.pgn

# 2. Process into training tensors (Black-only is the default)
python -m src.data.process_pgn

# 3. Train (writes models/kidoku_policy_best.pth, the best-validation checkpoint)
python train.py

# 4. Play against the bot (loads the best checkpoint; you play White)
python -m src.ui.play_pygame
```

Use `models/kidoku_policy_best.pth` for play and inference, not the final-epoch checkpoint:
the best-validation model generalizes better than the more memorized final one.

## Tech stack

Python, PyTorch, python-chess, Pygame.
## Special Thanks

Special thanks to **Vishak.P.Remesh** (kidokuismyname) for providing the game data used to train this bot.
