# Zombie Bridge Crossing Problem

## Problem Summary

This program implements a solution to the classic "Bridge and Torch" problem, reimagined as a zombie escape scenario. Four survivors (U, G, P, Pr) must cross a bridge at night, carrying a flashlight, while avoiding zombies. Only two survivors can cross at a time, and they must always have the flashlight. Each survivor moves at a different speed, and the goal is to find the fastest way for all to cross.

## Code Overview

The solution uses an informed search algorithm (A*) to find the optimal sequence of moves. The state space is encoded as numbered nodes, with dictionaries mapping possible transitions and their costs. The search avoids cycles and uses a heuristic based on the slowest remaining survivor.

### Files

- **zombie.py**  
  Contains the full implementation of the problem, including:
  - `Fringe`: A priority queue for managing the search frontier.
  - `Zombie_problem`: Encodes the state space, neighbor transitions, costs, and heuristic. Implements the A* search.
  - `unit_tests()`: Basic tests to verify correctness.
  - `main()`: Runs the tests and prints the solution path and cost.

## Assumptions

- The state space is fully encoded as integer nodes, with all possible transitions and costs precomputed.
- The heuristic is admissible and based on the maximum crossing time of survivors remaining on the starting side.
- The solution path avoids cycles by pruning backtracking moves.

## How to Run

From the `zombie-bridge-problem` directory, run:

```sh
python3 zombie.py
```

This will execute the unit tests and print the solution path and its total cost.