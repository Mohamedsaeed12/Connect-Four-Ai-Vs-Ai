# Adversarial Search Connect Four Assignment for ai vs ai
# Author: Mohamed Saeed and Jeremiah Roseman
# Date: 4/13/2025
# Purpose: This program implements a Connect Four game (6 rows x 5 columns) where a ai
# plays against an AI agent. it does an analysis and sees which versions of ai or which algothims are better 


import time
import math
import random
import copy

# Board dimensions and game constants
ROWS = 6
COLS = 5
HUMAN_PIECE = 1
AI_PIECE = 2
EMPTY = 0

# Global counters for performance metrics
nodes_visited = 0
max_depth_reached = 0

def create_board():
    """
    Create an empty Connect Four board.
    Returns a 6x5 grid where:
    - 0 = empty space
    - 1 = human piece
    - 2 = AI piece
    """
    # Create empty board
    board = []
    for row in range(ROWS):
        # Each row has 5 empty spaces
        row = [EMPTY] * COLS
        board.append(row)
    return board

def print_board(board): 
    
    # Print each row with cell contents
    for row in board:
        
        cells = []
        for cell in row:
            if cell == EMPTY:
                cells.append(' ')    # Empty space
            elif cell == HUMAN_PIECE:
                cells.append('1')    # Human piece
            else:
                cells.append('2')    # AI piece
        
        # Print row with separators
        print('| ' + ' | '.join(cells) + ' |')
 
    
    # Print column numbers at the bottom
    print('  1   2   3   4   5  ')  # Column numbers
    print('(Choose 1-5 to play)')    # Helpful reminder

def is_valid_move(board, col):
    """
    Check if a move in this column is allowed.
    Returns True if the top cell in the column is empty.
    """
    return board[0][col] == EMPTY

def get_valid_moves(board):
    """
    Get a list of all columns where a move is possible.
    Returns a list of column numbers (0-4) that aren't full.
    """
    valid_columns = []
    for col in range(COLS):
        if is_valid_move(board, col):
            valid_columns.append(col)
    return valid_columns

def make_move(board, col, piece):
    """
    Drop a piece into the chosen column.
    The piece falls to the lowest empty spot in that column.
    
    Example:
    Before:     After:
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | | | | |
    | | | | | |  | | |2| | |
    """
    # Make a copy of the board so we don't change the original
    new_board = copy.deepcopy(board)
    #deepcopy is a copy whose properties do not share the same references 
    #(point to the same underlying values) as those of the source object from which the copy was made
    for row in range(ROWS-1, -1, -1): #start at bottom and goes up5
        # If we find an empty cell in this column
        if new_board[row][col] == EMPTY:
            new_board[row][col] = piece
            break
    
    return new_board

def check_win(board, piece):
    """
    Check if a player has won by getting 4 pieces in a row.
    Checks in all directions: horizontal, vertical, and both diagonals.
    """
    # Check horizontal (left to right)
    for row in range(ROWS):
        for col in range(COLS - 3):
            # Check 4 cells in a row
            if (board[row][col] == piece and 
                board[row][col+1] == piece and 
                board[row][col+2] == piece and 
                board[row][col+3] == piece):
                return True

    # Check vertical (top to bottom)
    for col in range(COLS):
        for row in range(ROWS - 3):
            # Check 4 cells in a column
            if (board[row][col] == piece and 
                board[row+1][col] == piece and 
                board[row+2][col] == piece and 
                board[row+3][col] == piece):
                return True

    # Check diagonal (top-left to bottom-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            # Check 4 cells diagonally down-right
            if (board[row][col] == piece and 
                board[row+1][col+1] == piece and 
                board[row+2][col+2] == piece and 
                board[row+3][col+3] == piece):
                return True

    # Check diagonal (bottom-left to top-right)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            # Check 4 cells diagonally up-right
            if (board[row][col] == piece and 
                board[row-1][col+1] == piece and 
                board[row-2][col+2] == piece and 
                board[row-3][col+3] == piece):
                return True

    return False

def check_tie(board):
    """Return True if the board is full and no moves remain."""
    return all(board[0][col] != EMPTY for col in range(COLS))

def evaluate_window(window, piece):
    """
    Simple window evaluation that scores based on piece counts.
    """
    score = 0
    opponent_piece = HUMAN_PIECE if piece == AI_PIECE else AI_PIECE
    
    # Count pieces in window
    piece_count = window.count(piece)
    opponent_count = window.count(opponent_piece)
    empty_count = window.count(EMPTY)
    
    # Simple scoring:
    # - 1000 points for 4 in a row
    # - 100 points for 3 in a row with empty space
    # - 10 points for 2 in a row with empty spaces
    # - Block opponent's 3 or 2 in a row
    if piece_count == 4:
        score += 1000
    elif piece_count == 3 and empty_count == 1:
        score += 50
    elif piece_count == 2 and empty_count == 2:
        score += 10
    
    # Block opponent's threats
    if opponent_count == 3 and empty_count == 1:
        score += 40   # Block opponent's winning move
    elif opponent_count == 2 and empty_count == 2:
        score += 5    # Block opponent's potential
    
    return score

def evaluate_board(board, piece):
    """
    Simple board evaluation that:
    1. Gives bonus for center control
    2. Checks all possible 4-in-a-row windows
    3. Returns total score
    """
    score = 0
    
    # Center control bonus
    center_col = COLS // 2
    center_array = [board[row][center_col] for row in range(ROWS)]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    # Check all possible 4-in-a-row windows
    # Horizontal
    for row in range(ROWS):
        for col in range(COLS - 3):
            window = board[row][col:col+4]
            score += evaluate_window(window, piece)
    
    # Vertical
    for col in range(COLS):
        col_array = [board[row][col] for row in range(ROWS)]
        for row in range(ROWS - 3):
            window = col_array[row:row+4]
            score += evaluate_window(window, piece)
    
    # Diagonal (down-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            window = [board[row+i][col+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Diagonal (up-right)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            window = [board[row-i][col+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    return score

def is_terminal_node(board):
    """
    Check if the game is over.
    Returns True if:
    - Human player won
    - AI player won
    - Board is full (tie)
    """
    return (check_win(board, HUMAN_PIECE) or 
            check_win(board, AI_PIECE) or 
            check_tie(board))

def order_moves(board, valid_moves, piece):
    """
    Order moves based on their potential value to improve alpha-beta pruning efficiency.
    """
    move_scores = []
    for move in valid_moves:
        # Make a temporary move
        temp_board = make_move(board, move, piece)
        # Score the move based on the evaluation function
        score = evaluate_board(temp_board, piece)
        move_scores.append((score, move))
    
    # Sort moves by score in descending order
    move_scores.sort(reverse=True)
    return [move for _, move in move_scores]

def minimax(board, depth, alpha, beta, maximizingPlayer, start_time, time_limit, version):
    """
    Find the best move using the minimax algorithm.
    Version 1 uses alpha-beta pruning in second matchup, version 2 uses pure minimax.
    """
    global nodes_visited, max_depth_reached
    nodes_visited += 1
    max_depth_reached = max(max_depth_reached, depth)
    
    # Check if time limit exceeded
    if time.time() - start_time > time_limit:
        return evaluate_board(board, AI_PIECE), None
    
    # Check if terminal node or max depth reached
    if depth == 0 or is_terminal_node(board):
        if check_win(board, AI_PIECE):
            return (math.inf, None)
        elif check_win(board, HUMAN_PIECE):
            return (-math.inf, None)
        else:
            return (evaluate_board(board, AI_PIECE), None)
    
    # Get valid moves
    valid_moves = get_valid_moves(board)
    
    # Order moves for better pruning (only for AI Version 1 in second matchup)
    if version == 1 and match == 1:
        valid_moves = order_moves(board, valid_moves, AI_PIECE if maximizingPlayer else HUMAN_PIECE)
    
    if maximizingPlayer:
        value = -math.inf
        best_move = valid_moves[0]
        
        for move in valid_moves:
            # Make move and evaluate
            child_board = make_move(board, move, AI_PIECE)
            if version == 1 and match == 1:  # AI1 uses alpha-beta only in second matchup
                new_score, _ = minimax(child_board, depth-1, alpha, beta, False, start_time, time_limit, version)
                if new_score > value:
                    value = new_score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            else:  # Pure minimax for all other cases
                new_score, _ = minimax(child_board, depth-1, -math.inf, math.inf, False, start_time, time_limit, version)
                if new_score > value:
                    value = new_score
                    best_move = move
        return value, best_move
    else:
        value = math.inf
        best_move = valid_moves[0]
        for move in valid_moves:
            # Make move and evaluate
            child_board = make_move(board, move, HUMAN_PIECE)
            if version == 1 and match == 1:  # AI1 uses alpha-beta only in second matchup
                new_score, _ = minimax(child_board, depth-1, alpha, beta, True, start_time, time_limit, version)
                if new_score < value:
                    value = new_score
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            else:  # Pure minimax for all other cases
                new_score, _ = minimax(child_board, depth-1, -math.inf, math.inf, True, start_time, time_limit, version)
                if new_score < value:
                    value = new_score
                    best_move = move
        return value, best_move

def simple_ai_move(board):
    """
    Simple AI strategy that:
    1. Takes winning moves
    2. Blocks opponent's winning moves
    3. Otherwise plays randomly
    Returns: (move, nodes_visited, max_depth, time_used)
    """
    start_time = time.time()
    
    # Check for winning moves
    for col in range(COLS):
        if is_valid_move(board, col):
            temp_board = make_move(board, col, AI_PIECE)
            if check_win(temp_board, AI_PIECE):
                return col, 1, 1, time.time() - start_time
    
    # Check for opponent's winning moves
    for col in range(COLS):
        if is_valid_move(board, col):
            temp_board = make_move(board, col, HUMAN_PIECE)
            if check_win(temp_board, HUMAN_PIECE):
                return col, 1, 1, time.time() - start_time
    
    # Play randomly
    valid_moves = get_valid_moves(board)
    return random.choice(valid_moves), 1, 1, time.time() - start_time

def get_ai_move(board, version, time_limit):
    """
    Get the AI's move based on the version.
    Version 1: Simple AI in first matchup, minimax with alpha-beta in second
    Version 2: Always uses minimax without alpha-beta
    Both versions use exactly the same time limit.
    Returns: (move, nodes_visited, max_depth, time_used)
    """
    global nodes_visited, max_depth_reached
    
    if version == 1:
        # Use simple AI for first matchup, minimax with alpha-beta for second
        if match == 0:  # First matchup
            return simple_ai_move(board)
        else:  # Second matchup - use minimax with alpha-beta
            start_time = time.time()
            best_move = None
            depth = 1
            nodes_visited = 0
            max_depth_reached = 0

            while True:
                if time.time() - start_time > time_limit:
                    break
                    
                # Initialize alpha and beta for alpha-beta pruning
                score, move = minimax(board, depth, -math.inf, math.inf, True, start_time, time_limit, version)
                
                if time.time() - start_time < time_limit:
                    best_move = move
                    
                depth += 1
            
            total_time = time.time() - start_time
            if best_move is None:
                valid_moves = get_valid_moves(board)
                best_move = random.choice(valid_moves)
            
            return best_move, nodes_visited, max_depth_reached, total_time
    else:  # Version 2 - use minimax without alpha-beta
        start_time = time.time()
        best_move = None
        depth = 1
        nodes_visited = 0
        max_depth_reached = 0

        while True:
            if time.time() - start_time > time_limit:
                break
                
            score, move = minimax(board, depth, -math.inf, math.inf, True, start_time, time_limit, version)
            
            if time.time() - start_time < time_limit:
                best_move = move
                
            depth += 1
        
        total_time = time.time() - start_time
        if best_move is None:
            valid_moves = get_valid_moves(board)
            best_move = random.choice(valid_moves)
        
        return best_move, nodes_visited, max_depth_reached, total_time

def ai_vs_ai_game(version1, version2, time_limit=0.03):
    """
    Play a game between two AI versions.
    Returns: (winner, ai1_stats, ai2_stats)
    where ai_stats is (nodes_visited, max_depth, time_used)
    """
    board = create_board()
    game_over = False
    turn = 0
    
    # Track stats for both AIs
    ai1_total_nodes = 0
    ai1_max_depth = 0
    ai1_total_time = 0
    ai2_total_nodes = 0
    ai2_max_depth = 0
    ai2_total_time = 0
    
    while not game_over:
        # AI 1's turn
        if turn % 2 == 0:
            move, nodes, depth, time_used = get_ai_move(board, version1, time_limit)
            ai1_total_nodes += nodes
            ai1_max_depth = max(ai1_max_depth, depth)
            ai1_total_time += time_used
            
            if is_valid_move(board, move):
                board = make_move(board, move, AI_PIECE)
                if check_win(board, AI_PIECE):
                    return 1, (ai1_total_nodes, ai1_max_depth, ai1_total_time), (ai2_total_nodes, ai2_max_depth, ai2_total_time)
        # AI 2's turn
        else:
            move, nodes, depth, time_used = get_ai_move(board, version2, time_limit)
            ai2_total_nodes += nodes
            ai2_max_depth = max(ai2_max_depth, depth)
            ai2_total_time += time_used
            
            if is_valid_move(board, move):
                board = make_move(board, move, HUMAN_PIECE)
                if check_win(board, HUMAN_PIECE):
                    return 2, (ai1_total_nodes, ai1_max_depth, ai1_total_time), (ai2_total_nodes, ai2_max_depth, ai2_total_time)
        
        turn += 1
        
        # Check for tie
        if len(get_valid_moves(board)) == 0:
            return 0, (ai1_total_nodes, ai1_max_depth, ai1_total_time), (ai2_total_nodes, ai2_max_depth, ai2_total_time)
    
    return 0, (ai1_total_nodes, ai1_max_depth, ai1_total_time), (ai2_total_nodes, ai2_max_depth, ai2_total_time)

def run_ai_analysis(num_games=100):
    """
    Run two separate analyses of AI versions playing against each other with random starting order.
    Match 1: AI1 (Simple) vs AI2 (Minimax without Alpha-Beta)
    Match 2: AI1 (Minimax with Alpha-Beta) vs AI2 (Minimax only)
    Both AIs use exactly the same time limit per move.
    """
    print("\nRunning AI Analysis...")
    global match  # Make match a global variable
    
    # Set a fixed time limit for all AI moves
    time_limit = 0.03  # 30 milliseconds per move
    
    # Run two separate sets of games
    for match in range(2):
        print(f"\nMatch {match + 1}: {num_games} games with random starting order")
        print(f"Time limit per move: {time_limit*1000:.0f} milliseconds")
        if match == 0:
            print("AI Version 1: Simple Strategy")
            print("AI Version 2: Minimax without Alpha-Beta")
        else:
            print("AI Version 1: Minimax with Alpha-Beta")
            print("AI Version 2: Minimax without Alpha-Beta")
        
        # Statistics tracking
        wins_v1 = 0
        wins_v2 = 0
        ties = 0
        v1_total_nodes = 0
        v1_total_depth = 0
        v1_total_time = 0
        v2_total_nodes = 0
        v2_total_depth = 0
        v2_total_time = 0
        
        for game in range(num_games):
            print(f"\rPlaying game {game + 1}/{num_games}...", end="")
            
            # Randomly decide who goes first
            if random.random() < 0.5:
                winner, ai1_stats, ai2_stats = ai_vs_ai_game(1, 2, time_limit)
                if winner == 1:
                    wins_v1 += 1
                elif winner == 2:
                    wins_v2 += 1
                else:
                    ties += 1
                v1_total_nodes += ai1_stats[0]
                v1_total_depth = max(v1_total_depth, ai1_stats[1])
                v1_total_time += ai1_stats[2]
                v2_total_nodes += ai2_stats[0]
                v2_total_depth = max(v2_total_depth, ai2_stats[1])
                v2_total_time += ai2_stats[2]
            else:
                winner, ai2_stats, ai1_stats = ai_vs_ai_game(2, 1, time_limit)
                if winner == 1:
                    wins_v2 += 1
                elif winner == 2:
                    wins_v1 += 1
                else:
                    ties += 1
                v2_total_nodes += ai2_stats[0]
                v2_total_depth = max(v2_total_depth, ai2_stats[1])
                v2_total_time += ai2_stats[2]
                v1_total_nodes += ai1_stats[0]
                v1_total_depth = max(v1_total_depth, ai1_stats[1])
                v1_total_time += ai1_stats[2]
        
        print("\n\nAnalysis Results for Match", match + 1)
        print("=============================")
        print(f"Games played: {num_games}")
        print(f"AI Version 1 wins: {wins_v1} ({wins_v1/num_games*100:.1f}%)")
        print(f"AI Version 2 wins: {wins_v2} ({wins_v2/num_games*100:.1f}%)")
        print(f"Ties: {ties} ({ties/num_games*100:.1f}%)")
        print("\nPerformance Metrics:")
        print("===================")
        print("AI Version 1:")
        print(f"  Avg nodes per game: {v1_total_nodes/num_games:.0f}")
        print(f"  Max depth reached: {v1_total_depth}")
        print(f"  Avg time per move: {v1_total_time/num_games*1000:.1f} ms")
        
        print("\nAI Version 2:")
        print(f"  Avg nodes per game: {v2_total_nodes/num_games:.0f}")
        print(f"  Max depth reached: {v2_total_depth}")
        print(f"  Avg time per move: {v2_total_time/num_games*1000:.1f} ms")

def main():
    """
    Run the Connect Four AI analysis.
    """
    print("Welcome to Connect Four AI Analysis!")
    print("Board size: 6 rows x 5 columns. Get 4 in a row to win!")
    
    # Run AI analysis
    run_ai_analysis()

if __name__ == "__main__":
    main()
