# Sudoku Solver

## Overview

This project implements a Sudoku solver using constraint propagation and search techniques. The solver can handle standard 9x9 Sudoku puzzles, including challenging instances from benchmark sets. The code was developed to showcase constraint satisfaction problems (CSPs) and efficient backtracking algorithms.

## Features

- **Constraint Propagation:** Applies techniques such as elimination and only-choice to reduce the search space.
- **Backtracking Search:** Uses depth-first search to explore possible assignments when constraint propagation alone is insufficient.
- **Benchmark Support:** Includes hard puzzles from `top95.txt` and a tutorial example in `tutorial_problem.txt`.
- **Performance Measurement:** Records and visualizes running time for solving different puzzles.

## File Structure

- main.py: Main Sudoku solver implementation.
- requirements.txt: Python dependencies.
- top95.txt: Benchmark set of hard Sudoku puzzles.
- tutorial_problem.txt: Example puzzle for testing.
- running_time.png: Visualization of solver performance.

## Assumptions

- Input puzzles are provided as plain text files with one puzzle per line (e.g., `top95.txt`).
- The solver is designed for standard 9x9 Sudoku puzzles.
- Python 3 is required, and dependencies are listed in `requirements.txt`.

## How to Run

1. **Install Dependencies**

   From the `sudoku-solver/` directory, install the required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Solver** 
To solve the tutorial problem or run on benchmark puzzles, execute:

   ```sh
   python sudoku_solver.py
   ```

By default, the script will:
- Load a sample puzzle from `tutorial_problem.txt`.
- Solve the puzzle and print the solution.
- Optionally, process all puzzles in `top95.txt` and report statistics.

3. **View Results**
   The solver will output the solved puzzle to the console. Running time statistics are visualized in running_time.png.