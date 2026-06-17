# Kidoku Bot

A neural network chess bot trained to **play like "kidokuismyname"** (~500-600 ELO on Chess.com) using behavioral cloning on their real game history.

## Goal

Create an AI that captures the playing style, mistakes, opening preferences, and personality of a specific low-rated human player — not to play strong chess, but to feel *human*.

## Why this is interesting

Most chess engines play perfectly or near-perfectly. This project goes the opposite direction: it tries to replicate the beautiful imperfection of human play at the ~600 ELO level.

## Current Status

🚧 **Early development** — Starting fresh with a clean, proper implementation.

### Key improvements over previous attempts:
- Fixed large policy head (can suggest *any* legal move, not limited to 492 seen moves)
- Proper data pipeline
- Modern but lightweight ResNet-style architecture
- Temperature-controlled sampling for human-like variability
- Clean, maintainable codebase

## Project Structure

```
kidoku-bot/
├── data/
│   ├── raw/              # Original PGN files from Chess.com
│   └── processed/        # Tokenized boards + move targets
├── src/
│   ├── data/             # Data loading & preprocessing
│   ├── model/            # Neural network architectures
│   ├── train/            # Training loops, loss, logging
│   ├── inference/        # get_bot_move() with temperature, mate detection, etc.
│   └── ui/               # Pygame interface + future Streamlit
├── models/               # Saved weights (.pth)
├── notebooks/            # Exploration & analysis
├── train.py              # Main training entrypoint
├── play.py               # Play against the bot (Pygame)
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Clean data pipeline (PGN → training examples)
- [ ] Fixed policy head architecture (from-to + promotion)
- [ ] Training script with proper logging & checkpoints
- [ ] Strong inference engine (temperature, mate-in-1, legal move masking)
- [ ] Nice Pygame UI (drag & drop, move history, evaluation feel)
- [ ] Evaluation: How well does it imitate Kidoku? (move prediction accuracy + human Turing test)
- [ ] Optional: Opening book + simple search (if we want to tune strength later)

## Installation

```bash
git clone https://github.com/yourname/kidoku-bot.git
cd kidoku-bot
pip install -r requirements.txt
```

## Quick Start (once trained)

```bash
python play.py
```

## Data

The model is trained on real games played by **kidokuismyname** on Chess.com (rapid 10+0).

If you want to train your own "clone" of a different player, just replace the PGN file in `data/raw/`.

## Philosophy

> "The goal is not to beat the player. The goal is to *become* the player."

This project prioritizes **style and personality** over Elo rating.

## License

MIT

## Acknowledgments

- Built with `python-chess`, PyTorch, and love for imperfect human chess.
- Inspired by projects like Maia Chess, but focused on much lower-rated play.

---

**Want to follow the development?** Star the repo and watch the commits. We'll build this properly, one clean step at a time.
