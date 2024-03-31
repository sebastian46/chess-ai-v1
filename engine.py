import chess

# Define piece values
piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}

def evaluate_board_material(board):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.symbol().lower()]
            if piece.color == chess.BLACK:
                score += value
            else:
                score -= value
    return score

def select_best_move(board, legal_moves):
    best_move = None
    best_score = float('-inf')
    for move in legal_moves:
        board.push(move)
        score = evaluate_board_material(board)
        board.pop()
        if score > best_score:
            best_move = move
            best_score = score
    return best_move
