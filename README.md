# Kidoku Bot

A neural network that learns to play chess like **kidokuismyname** (~500-600 ELO) using behavioral cloning on real game data.

## Goal

Build a bot that captures the playing style, opening preferences, and typical mistakes of a specific low-rated human player rather than playing optimally.

## Current Status

- Data processing pipeline (PGN → training tensors)
- `KidokuPolicyNet` (ResNet-style policy network)
- Training script with checkpointing
- Inference engine with temperature sampling
- Model trained for 32 epochs on ~50 games

**Note:** The current model was trained on a relatively small dataset (50 games). It shows high training accuracy but may be overfitting. More games will be added in future training runs.

## Project Structure
kidoku-bot/
├── data/
│   ├── raw/                  # Original PGN files
│   └── processed/            # Training tensors (.pt files)
├── src/
│   ├── data/                 # PGN processing
│   ├── model/                # KidokuPolicyNet
│   ├── train/                # Training loop
│   └── inference/            # KidokuBot inference class
├── models/                   # Saved model weights (.pth)
├── train.py                  # Main training entrypoint
└── README.md
pip install -r requirements.txt
python train.py
Training checkpoints are saved in the models/ folder after every epoch.
How to Use the Bot (Inference)
Pythonfrom src.inference import KidokuBot
import chess

bot = KidokuBot(model_path="models/kidoku_policy_final.pth", temperature=1.6)

board = chess.Board()
move = bot.get_move(board)
print(move)   # Returns UCI move (e.g. "e2e4")
Higher temperature = more random/human-like moves.
Lower temperature = more deterministic.
Limitations (Current)

Trained on limited data (~50 games)
High training accuracy likely due to overfitting
No search / value head (pure policy model)
Best used as a style imitation experiment rather than a strong player

Future Plans

Train on significantly more games
Add validation split to monitor overfitting
Build playable Pygame interface
Experiment with temperature tuning and move selection

Acknowledgments
Built using python-chess and PyTorch.
