import pygame
import chess
import os
import random
import engine
from tensorflow.keras.models import load_model
import a0
import numpy as np

model = load_model('models/v1.h5')
def model_move(board, model):
    """Make a move based on the model's prediction."""
    if board.is_game_over():
        # No move to make if the game is over
        return None

    board_input = a0.board_to_input(board).reshape(1, 8, 8, 14)  # Adjust the shape for the model input
    policy, _ = model.predict(board_input)
    legal_moves = list(board.legal_moves)
    move_probs = np.zeros(len(legal_moves))
    
    if not legal_moves:
        # If there are no legal moves, return None
        return None

    for i, move in enumerate(legal_moves):
        move_index = a0.move_to_index(move)
        move_probs[i] = policy[0, move_index]
    # Select the move with the highest probability
    best_move = legal_moves[np.argmax(move_probs)]
    return best_move

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_size = 600
square_size = screen_size // 8
coordinates_size = 20  # Set a fixed size for the coordinates margin
screen = pygame.display.set_mode((screen_size + 2 * coordinates_size, screen_size + 2 * coordinates_size))

# Set up the font for drawing text
font = pygame.font.SysFont('Arial', coordinates_size)

# Load images for pieces (assuming you have PNG images for each piece)
piece_images = {}
pieces = ['p', 'n', 'b', 'r', 'q', 'k']
for piece in pieces:
    # Load white pieces
    piece_images['w_' + piece.upper()] = pygame.transform.scale(pygame.image.load(os.path.join('images', f'w_{piece}.png')), (square_size, square_size))
    # Load black pieces
    piece_images['b_' + piece] = pygame.transform.scale(pygame.image.load(os.path.join('images', f'b_{piece}.png')), (square_size, square_size))

def draw_board(board):
    # Draw the board squares
    for r in range(8):
        for c in range(8):
            color = pygame.Color("white") if (r + c) % 2 == 0 else pygame.Color("gray")
            pygame.draw.rect(screen, color, pygame.Rect(c*square_size + coordinates_size, r*square_size + coordinates_size, square_size, square_size))
            
    # Draw the pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            column = square % 8
            row = 7 - square // 8
            color_prefix = 'w_' if piece.color == chess.WHITE else 'b_'
            piece_key = color_prefix + str(piece)
            screen.blit(piece_images[piece_key], (column*square_size + coordinates_size, row*square_size + coordinates_size))

def draw_coordinates():
    # Draw the coordinates around the board
    for i in range(8):
        # Files (letters)
        file_surface = font.render(chr(ord('a') + i), True, pygame.Color('black'))
        screen.blit(file_surface, (coordinates_size + i * square_size + square_size//2 - file_surface.get_width()//2, screen_size + coordinates_size + coordinates_size//2 - file_surface.get_height()//2))
        # Ranks (numbers)
        rank_surface = font.render(str(8-i), True, pygame.Color('black'))
        screen.blit(rank_surface, (coordinates_size//2 - rank_surface.get_width()//2, i * square_size + coordinates_size + square_size//2 - rank_surface.get_height()//2))

# Function to convert pixel positions to chess square
def pixel_to_square(x, y):
    file = (x - coordinates_size) // square_size
    rank = 7 - (y - coordinates_size) // square_size
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)

# Function to draw a message on the screen
def draw_message(message, color, position):
    font = pygame.font.SysFont('Arial', 36)
    text_surface = font.render(message, True, pygame.Color(color))
    rect = text_surface.get_rect()
    rect.center = position
    screen.blit(text_surface, rect)

# Function to highlight a square
def highlight_square(screen, square, color):
    pygame.draw.rect(screen, color, pygame.Rect((square % 8) * square_size + coordinates_size, 
                                                (7 - square // 8) * square_size + coordinates_size, 
                                                square_size, square_size), 5)

# Function to draw circles for legal moves
def draw_legal_moves(screen, board, square):
    for move in board.legal_moves:
        if move.from_square == square:
            center_x = (move.to_square % 8) * square_size + square_size // 2 + coordinates_size
            center_y = (7 - move.to_square // 8) * square_size + square_size // 2 + coordinates_size
            pygame.draw.circle(screen, pygame.Color("blue"), (center_x, center_y), square_size // 10)

selected_piece_square = None

# Game loop
running = True
board = chess.Board()
while running:
    is_checkmate = board.is_checkmate()
    is_stalemate = board.is_stalemate()
    is_game_over = is_checkmate or is_stalemate

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if board.turn == chess.WHITE and not is_game_over:
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                clicked_square = pixel_to_square(x, y)
                if clicked_square is not None:
                    if selected_piece_square:
                        move = chess.Move(selected_piece_square, clicked_square)
                        if move in board.legal_moves:
                            board.push(move)
                            selected_piece_square = None
                        else:
                            # Invalid move, deselect the piece
                            selected_piece_square = None
                    elif board.piece_at(clicked_square) and board.color_at(clicked_square) == chess.WHITE:
                        # Select the piece on the clicked square
                        selected_piece_square = clicked_square

    if board.turn == chess.BLACK and not is_game_over:
        # It's the computer's turn to move (as black)
        # legal_moves = list(board.legal_moves)
        # if legal_moves:
            # best_move = engine.select_best_move(board, legal_moves)
            # board.push(best_move)
            # Deselect piece after move
            # selected_piece_square = None
        ai_move = model_move(board, model)
        if ai_move:
            board.push(ai_move)
        selected_piece_square = None

    screen.fill(pygame.Color("white"))
    draw_board(board)
    draw_coordinates()

    # Check for endgame and draw message
    if is_checkmate:
        draw_message("Checkmate!", "red", (screen_size // 2, screen_size // 2))
    elif is_stalemate:
        draw_message("Stalemate!", "blue", (screen_size // 2, screen_size // 2))

    # Highlight selected piece and legal moves
    if selected_piece_square and not is_game_over:
        highlight_square(screen, selected_piece_square, pygame.Color("green"))
        draw_legal_moves(screen, board, selected_piece_square)

    pygame.display.flip()

pygame.quit()
