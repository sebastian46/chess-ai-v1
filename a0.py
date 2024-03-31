import tensorflow as tf
from tensorflow.keras import layers, models
import chess
import numpy as np
import math
import random

def create_chess_model():
    input_shape = (8, 8, 14)  # 8x8 board, 14 channels (6 pieces types * 2 colors + 2 for castling rights for each color)
    inputs = layers.Input(shape=input_shape)

    # Convolutional block
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Residual blocks
    for _ in range(5):
        skip = x
        x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([skip, x])
        x = layers.ReLU()(x)

    # Policy head
    policy_conv = layers.Conv2D(2, kernel_size=1, activation="relu")(x)
    policy_conv = layers.BatchNormalization()(policy_conv)
    policy_flat = layers.Flatten()(policy_conv)
    policy_output = layers.Dense(4672, activation="softmax", name="policy_output")(policy_flat)  # 4672 possible move positions in chess

    # Value head
    value_conv = layers.Conv2D(1, kernel_size=1)(x)
    value_conv = layers.BatchNormalization()(value_conv)
    value_flat = layers.Flatten()(value_conv)
    value_hidden = layers.Dense(64, activation="relu")(value_flat)
    value_output = layers.Dense(1, activation="tanh", name="value_output")(value_hidden)  # Win probability [-1, 1]

    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
    
    return model

def board_to_input(board):
    """
    Converts a chess.Board object to a 3D numpy array (8, 8, 14).
    - 12 channels for the 6 types of pieces, each for white and black.
    - 2 additional channels for castling rights (white, black).
    """
    piece_to_channel = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    board_state = np.zeros((8, 8, 14), dtype=np.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            channel = piece_to_channel[piece.piece_type]
            if piece.color == chess.WHITE:
                board_state[row, col, channel] = 1
            else:
                board_state[row, col, channel + 6] = 1
    
    # Castling rights
    board_state[:, :, 12] = board.has_kingside_castling_rights(chess.WHITE)
    board_state[:, :, 13] = board.has_kingside_castling_rights(chess.BLACK)
    
    # Possible enhancement: add channel for en-passant square
    
    return board_state

def simulate_game_data(model, board):
    """Simulate a single game's data for demonstration."""
    moves = []
    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        move = np.random.choice(legal_moves)
        board.push(move)
        moves.append(move)
    return moves

class Node:
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value_sum = 0

class MCTS:
    def __init__(self, model, num_simulations, c_puct=1.0, batch_size=32):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
    
    def search(self, board):
        root = Node(board)
        
        for i in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Select
            while node.children:
                node = self.select_child(node)
                search_path.append(node)
            
            # Expand
            if not node.board.is_game_over():
                policy, value = self.predict_batch(search_path)  # Predict in batch
                self.expand_node(node, policy)
            else:
                value = self.get_outcome_value(node.board)
            
            self.backpropagate(search_path, value)
        
        return self.get_best_move(root)
    
    def predict_batch(self, search_path):
        batch_size = self.batch_size
        num_batches = (len(search_path) + batch_size - 1) // batch_size
        
        policies = []
        values = []
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(search_path))
            
            batch = search_path[start:end]
            batch_input = np.array([board_to_input(node.board) for node in batch])
            
            batch_policy, batch_value = self.model.predict(batch_input)
            
            policies.extend(batch_policy)
            values.extend(batch_value[:, 0])
        
        return policies[-1], values[-1]

    
    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children)

        if total_visits == 0:
            return random.choice(node.children)

        log_total_visits = math.log(total_visits)
        
        best_score = -1
        best_child = None
        
        for child in node.children:
            visits = child.visits + 1e-8  # Add a small constant to avoid division by zero
            score = child.value_sum / visits + self.c_puct * child.prior * math.sqrt(log_total_visits) / (1 + visits)
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand_node(self, node, policy):
        for move in node.board.legal_moves:
            new_board = node.board.copy()
            new_board.push(move)
            prior = policy[move_to_index(move)]
            child_node = Node(new_board, parent=node, move=move, prior=prior)
            node.children.append(child_node)
    
    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += value
            value = -value
    
    def get_best_move(self, root):
        best_visits = -1
        best_move = None
        
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move
        
        return best_move
    
    def predict(self, board):
        board_input = board_to_input(board)
        policy, value = self.model.predict(np.expand_dims(board_input, axis=0))
        policy = policy.flatten()
        value = value[0][0]
        return policy, value
    
    def get_outcome_value(self, board):
        outcome = board.outcome()
        if outcome.winner is None:
            return 0
        elif outcome.winner == chess.WHITE:
            return 1
        else:
            return -1

def generate_self_play_data(model, mcts, games=1, temperature=1.0):
    data = []
    for game_idx in range(games):
        print(f"Starting game {game_idx + 1}")
        board = chess.Board()
        game_history = []
        move_count = 0
        while not board.is_game_over():
            move_count += 1
            # print(f"Move {move_count}")
            board_input = board_to_input(board)
            policy, value = model.predict(np.expand_dims(board_input, axis=0))
            policy = policy.flatten()
            # print("Selecting move")
            move = mcts.search(board)  # Use MCTS to select the move
            # print(f"Selected move: {move}")
            game_history.append((board_input, policy, value))
            board.push(move)
        # print("Game over")
        outcome = board.outcome()
        # Update the game history with the final outcome
        for i, (board_input, policy, value) in enumerate(game_history):
            if outcome.winner is None:
                result = 0
            elif outcome.winner == chess.WHITE:
                result = 1
            else:
                result = -1
            data.append((board_input, policy, result))
    return data

def preprocess_data(data):
    # Convert data to a format suitable for training: features, policy_targets, value_targets
    features = np.array([item[0] for item in data])
    policy_targets = np.array([item[1] for item in data])
    value_targets = np.array([item[2] for item in data]).reshape(-1, 1)  # Reshape for Keras
    return features, policy_targets, value_targets

def train_model(model, data):
    # Pseudocode for training the model on generated data
    # 1. Preprocess the data into a suitable format for the model
    # 2. Separate the data into features (board states) and labels (moves, outcomes)
    # 3. Train the model on the data
    features, policy_targets, value_targets = preprocess_data(data)
    
    model.compile(optimizer='adam', 
                  loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
                  metrics={'policy_output': 'accuracy', 'value_output': 'mae'})

    model.fit(features, {'policy_output': policy_targets, 'value_output': value_targets}, epochs=10)

def move_to_index(move):
    """
    Converts a chess move into an index in the range [0, 4671].
    This accounts for regular moves and promotions.
    """
    from_square = move.from_square
    to_square = move.to_square
    promotion_piece = 0  # Default to no promotion

    if move.promotion:
        # Map the promotion piece to a number (knight=1, bishop=2, rook=3, queen=4)
        promotion_piece = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]

    # Calculate the index
    index = from_square * 64 + to_square

    # Adjust the index for promotions, which are stored after the first 4096 indices
    if promotion_piece > 0:
        index = 4096 + (from_square * 4 + (promotion_piece - 1))

    return index

def index_to_move(index, board):
    """
    Converts an index in the range [0, 4671] back into a chess move.
    """
    if index < 4096:
        from_square = index // 64
        to_square = index % 64
        promotion = None
    else:
        # Adjust for promotions
        index -= 4096
        from_square = index // 4
        to_square = from_square + ((index % 4) + 1) * 8  # Simplified, real calculation may vary
        promotion = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][index % 4]

    return chess.Move(from_square, to_square, promotion)

if __name__ == "__main__":
    model = create_chess_model()
    mcts = MCTS(model, num_simulations=5)  # Create an instance of MCTS
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'},
                  metrics={'policy_output': 'accuracy', 'value_output': 'mae'})
    
    for iteration in range(1):  # Run multiple iterations of self-play and training
        print(f"Starting iteration {iteration+1}")
        games_data = generate_self_play_data(model, mcts, games=5, temperature=1.0)
        print("Generated self-play data")
        features, policy_targets, value_targets = preprocess_data(games_data)
        print("Preprocessed data")
        
        history = model.fit(features, {'policy_output': policy_targets, 'value_output': value_targets}, 
                            epochs=10, batch_size=256, verbose=1)  # Increased batch size
        print("Model trained")
        
        print(f"Iteration {iteration+1}")
        print("Policy Loss:", history.history['policy_output_loss'][-1])
        print("Policy Accuracy:", history.history['policy_output_accuracy'][-1])
        print("Value Loss:", history.history['value_output_loss'][-1])
        print("Value MAE:", history.history['value_output_mae'][-1])
        print()
    
    model.save('models/v1.h5')
