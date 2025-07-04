# Connect Four AI vs AI Analysis

This project implements a Connect Four game (6 rows x 5 columns) where two AI agents play against each other. The goal is to analyze and compare different AI strategies and algorithms for Connect Four.

## Authors
- Mohamed Saeed
- Jeremiah Roseman

## Features
- Two AI versions:
  - **Version 1:** Minimax algorithm with alpha-beta pruning
  - **Version 2:** Minimax algorithm without alpha-beta pruning
- Randomized starting player for each game
- Performance analysis over 100 games (by default)
- Tracks wins, ties, nodes searched, maximum depth, and average move time

## How to Run
1. Make sure you have Python 3 installed.
2. Place all files in the same directory.
3. Run the main script:
   ```bash
   python AI_HW4.py
   ```
4. The program will automatically run 100 AI vs AI games and print analysis results to the console.

## Output Explanation
- **AI Version 1 wins:** Number and percentage of games won by the alpha-beta pruning AI
- **AI Version 2 wins:** Number and percentage of games won by the regular minimax AI
- **Ties:** Number and percentage of games that ended in a tie
- **Performance Metrics:**
  - *Average nodes searched per game*: How many board states each AI evaluated on average
  - *Maximum depth reached*: The deepest search level reached by each AI
  - *Average time per move*: How long, on average, each AI took to make a move (in milliseconds)

## Customization
- You can change the number of games or the time limit per move by editing the `run_ai_analysis` function in `AI_HW4.py`.

## License
This project is for educational purposes. 