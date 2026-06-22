# Kidoku Bot

A neural network chess bot trained to play like **kidokuismyname** (\~500-600 ELO) using behavioral cloning on real Chess.com games.

## Goal

Create a bot that captures the playing **style**, opening preferences, and typical mistakes of a specific low-rated human player, rather than playing optimally like a traditional chess engine.

## Current Status (as of June 2026)

- **Data Pipeline**: Converts PGN games into training tensors
- **Model**: `KidokuPolicyNet` — A ResNet-style convolutional neural network
- **Training**: Script with checkpointing (model trained up to 36+ epochs)
- **Inference**: `KidokuBot` class with temperature sampling for human-like play
- **UI**: Fully playable Pygame interface with:
  - Real chess piece images
  - Click-to-move
  - Legal move highlights
  - Move history sidebar
  - Undo, New Game, and Resign buttons

**Note**: The model was trained on a relatively small dataset (\~50 games). It currently shows signs of overfitting. We plan to improve it by training on significantly more games.

## Project Structure
kidoku-bot/
├── assets/
│   └── pieces/           # Chess piece images (PNG)
├── data/
│   ├── raw/              # Original PGN files
│   └── processed/        # Training data (.pt files)
├── models/               # Saved model weights (.pth)
├── src/
│   ├── data/             # PGN processing
│   ├── model/            # KidokuPolicyNet
│   ├── train/            # Training loop
│   ├── inference/        # KidokuBot inference
│   └── ui/               # Pygame interface
├── train.py
└── README.md
## How to Run

### 1. Play Against the Bot (Recommended)

```bash
python -m src.ui.play_pygame
This launches the full playable interface with your trained model.
2. Train the Model
python train.py
Training checkpoints are saved in the models/ folder.
3. Test Inference Directly
python -m src.inference.get_bot_move
Limitations
Currently trained on limited data (~50 games)
High training accuracy but likely overfitting
Bot can play "dumb" moves in some positions
No search or value head (pure policy model)
Future Plans
Train on significantly more games (200–500+)
Add validation split to monitor overfitting
Improve move selection and temperature tuning
Further UI improvements if needed
Tech Stack
Python
PyTorch
python-chess
Pygame
Built as a learning/experimentation project to explore behavioral cloning on human chess games.