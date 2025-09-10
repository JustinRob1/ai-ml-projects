# Connect Four: Minimax and Alpha-Beta Pruning

## Overview

This project implements AI agents for the game of Connect Four using the Minimax algorithm and Alpha-Beta pruning. The code demonstrates adversarial search and game tree evaluation techniques. The AI agents can play against each other or a human, demonstrating the effectiveness of search algorithms in two-player, zero-sum games.

## Features

- **Connect Four Game Logic:** Complete implementation of the Connect Four game, including move validation and win detection.
- **Minimax Agent:** Explores the game tree to select optimal moves based on minimax values.
- **Alpha-Beta Pruning Agent:** Optimized version of minimax that prunes branches to improve efficiency.
- **Testing Scripts:** Includes scripts to test and compare the performance of both algorithms.

## File Structure
- connect4.py: Main Connect Four game and AI agent implementations.
- testminimax.py: Script to test the Minimax agent.
- testalphabeta.py: Script to test the Alpha-Beta pruning agent.
- Assignment 3.pdf, assignment-3.pdf: Assignment description and requirements.

## Assumptions

- The game is played on a standard 7-column, 6-row Connect Four board.
- Python 3 is required to run the code.
- No external dependencies are needed beyond the Python standard library.

## How to Run

1. **Play or Test the AI Agents**

   - To run the Connect Four game or test the AI agents, use the provided scripts in the `c4-alpha-beta` directory.

2. **Run Minimax Tests**

   ```sh
   python testminimax.py
   ```

3. **Run Alpha-Beta Pruning Tests**

   ```sh
   python testalphabeta.py
    ```

### Game Logic
- The main game logic is implemented in `connect4.py`.
- You can modify or extend the code to play against the AI or to experiment with different evaluation functions.