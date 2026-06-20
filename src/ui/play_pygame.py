"""
Playable Pygame Interface for Kidoku Bot

Features:
- Real chess piece images
- Click to move
- Legal move highlights
- Move history sidebar
- Undo / New Game / Resign buttons
- Loads trained model by default
"""

import pygame
import chess
import sys
from pathlib import Path

from src.inference.get_bot_move import KidokuBot

# ==================== CONFIG ====================
BOARD_SIZE = 640
SQ_SIZE = BOARD_SIZE // 8
SIDEBAR_WIDTH = 280
WIDTH = BOARD_SIZE + SIDEBAR_WIDTH
HEIGHT = BOARD_SIZE

PIECE_PATH = Path("assets/pieces")

# Colors
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (130, 151, 105)
SELECTED = (246, 246, 105)
SIDEBAR_BG = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Kidoku Bot - Playable")
font = pygame.font.SysFont("Arial", 18)
font_small = pygame.font.SysFont("Arial", 14)

# Load piece images
def load_pieces():
    pieces = {}
    piece_map = {
        'P': 'wP.png', 'N': 'wN.png', 'B': 'wB.png',
        'R': 'wR.png', 'Q': 'wQ.png', 'K': 'wK.png',
        'p': 'bP.png', 'n': 'bN.png', 'b': 'bB.png',
        'r': 'bR.png', 'q': 'bQ.png', 'k': 'bK.png'
    }
    for symbol, filename in piece_map.items():
        img = pygame.image.load(PIECE_PATH / filename)
        pieces[symbol] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
    return pieces

PIECES = load_pieces()

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.move_history = []
        self.bot = KidokuBot(model_path="models/kidoku_policy_final.pth", temperature=1.6)
        self.game_over = False
        self.result = ""

    def reset(self):
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.move_history = []
        self.game_over = False
        self.result = ""

    def handle_click(self, pos):
        if self.game_over:
            return

        x, y = pos
        if x > BOARD_SIZE:
            return  # Clicked on sidebar

        file = x // SQ_SIZE
        rank = 7 - (y // SQ_SIZE)
        square = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move.uci())
                self.selected_square = None
                self.legal_moves = []

                # Bot plays
                if not self.board.is_game_over():
                    bot_move = self.bot.get_move(self.board)
                    self.board.push_uci(bot_move)
                    self.move_history.append(bot_move)
            else:
                self.selected_square = None
                self.legal_moves = []

    def undo(self):
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            if len(self.move_history) >= 2:
                self.move_history.pop()
                self.move_history.pop()
        self.selected_square = None
        self.legal_moves = []

    def resign(self):
        self.game_over = True
        self.result = "You resigned"

# ==================== DRAWING ====================

def draw_board(screen, game):
    for rank in range(8):
        for file in range(8):
            color = LIGHT if (rank + file) % 2 == 0 else DARK
            rect = pygame.Rect(file * SQ_SIZE, rank * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, color, rect)

            # Highlight selected square
            if game.selected_square == chess.square(file, 7 - rank):
                s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                s.fill((246, 246, 105, 150))
                screen.blit(s, rect.topleft)

            # Highlight legal moves
            sq = chess.square(file, 7 - rank)
            if sq in [m.to_square for m in game.legal_moves]:
                center = (file * SQ_SIZE + SQ_SIZE // 2, rank * SQ_SIZE + SQ_SIZE // 2)
                pygame.draw.circle(screen, (0, 0, 0, 80), center, SQ_SIZE // 6)

def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = square % 8
            rank = 7 - (square // 8)
            symbol = piece.symbol()
            screen.blit(PIECES[symbol], (file * SQ_SIZE, rank * SQ_SIZE))

def draw_sidebar(screen, game):
    sidebar_rect = pygame.Rect(BOARD_SIZE, 0, SIDEBAR_WIDTH, HEIGHT)
    pygame.draw.rect(screen, SIDEBAR_BG, sidebar_rect)

    # Title
    title = font.render("Move History", True, TEXT_COLOR)
    screen.blit(title, (BOARD_SIZE + 20, 20))

    # Move history
    y = 60
    for i, move in enumerate(game.move_history[-20:]):  # Show last 20 moves
        text = font_small.render(f"{i+1}. {move}", True, TEXT_COLOR)
        screen.blit(text, (BOARD_SIZE + 20, y))
        y += 22

    # Buttons
    button_y = HEIGHT - 120
    buttons = [
        ("Undo", (BOARD_SIZE + 20, button_y)),
        ("New Game", (BOARD_SIZE + 20, button_y + 40)),
        ("Resign", (BOARD_SIZE + 20, button_y + 80))
    ]

    for text, pos in buttons:
        rect = pygame.Rect(pos[0], pos[1], 200, 32)
        pygame.draw.rect(screen, (70, 70, 70), rect)
        label = font_small.render(text, True, TEXT_COLOR)
        screen.blit(label, (pos[0] + 60, pos[1] + 6))

def draw_game_over(screen, game):
    if game.game_over:
        overlay = pygame.Surface((BOARD_SIZE, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        text = font.render(game.result, True, (255, 100, 100))
        screen.blit(text, (BOARD_SIZE // 2 - 80, HEIGHT // 2))

# ==================== MAIN LOOP ====================

def main():
    game = Game()
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos

                    # Check sidebar buttons
                    if x > BOARD_SIZE:
                        button_y = HEIGHT - 120
                        if button_y <= y <= button_y + 32:
                            game.undo()
                        elif button_y + 40 <= y <= button_y + 72:
                            game.reset()
                        elif button_y + 80 <= y <= button_y + 112:
                            game.resign()
                    else:
                        game.handle_click(event.pos)

        screen.fill((0, 0, 0))
        draw_board(screen, game)
        draw_pieces(screen, game.board)
        draw_sidebar(screen, game)
        draw_game_over(screen, game)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
