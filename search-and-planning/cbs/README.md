# Conflict-Based Search for Multi-Agent Pathfinding

## Overview

This project implements the Conflict-Based Search (CBS) algorithm to solve the Multi-Agent Pathfinding (MAPF) problem. The MAPF problem involves finding collision-free paths for multiple agents from their respective start positions to their goals on a shared grid map, minimizing the total cost or makespan. The maps used are from the [movingai.org](https://movingai.com/benchmarks/grids.html) DAO benchmark set.

The code explores advanced search techniques for multi-agent systems. It demonstrates the use of CBS, a two-level search algorithm that resolves conflicts between agents in a principled and efficient way.

## Features

- **CBS Algorithm:** Efficiently finds optimal, conflict-free paths for multiple agents.
- **Flexible Map Support:** Works with a variety of grid maps in DAO format (see `dao-map/`).
- **Batch Testing:** Supports running on multiple test instances for benchmarking.
- **Statistics Reporting:** Outputs solution paths, costs, and correctness checks.

## Assumptions

- Map files are in the DAO format and located in the `dao-map/` directory.
- Test instances specifying agent start and goal positions are in the `test-instances/` directory.
- The program is written in Python 3 and requires the packages listed in `requirements.txt`.
- The main entry point is [`main.py`](main.py).

## How to Run

1. **Install Dependencies**

   From the `cbs/` directory, install the required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Program**

   To solve a sample problem or run batch tests, execute::

   ```sh
   python main.py
   ```

By default, the script will:
- Load a sample map and agent configuration.
- Run CBS to find paths for all agents.
- Print the solution paths and costs.
- Run through test instances in test-instances/ and check correctness.

You can modify the map or test instance files used by editing main.py.

### Example Output
```sh
Solution paths encountered for the easy test:
0 [State(1, 1), State(2, 1), State(3, 1), State(4, 1)]
1 [State(5, 1), State(4, 1), State(3, 1), State(2, 1)]

Correctly Solved:  8 8
```

