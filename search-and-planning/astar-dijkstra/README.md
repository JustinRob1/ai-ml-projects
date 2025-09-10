# A* and Dijkstra Pathfinding Algorithms

## Overview

This project implements the A* and Dijkstra search algorithms to solve pathfinding problems on grid-based maps. The code explores and compares heuristic and uninformed search strategies in artificial intelligence.

The program reads map files in the DAO format, representing environments with obstacles and varying terrain costs, and computes the shortest path between a start and goal position. It outputs the path found, the cost, and statistics such as the number of nodes expanded.

## Features

- **A\* Search:** Uses heuristics (typically Manhattan or Euclidean distance) to efficiently find the shortest path.
- **Dijkstra's Algorithm:** Finds the shortest path without using a heuristic (equivalent to A* with a zero heuristic).
- **Flexible Map Support:** Works with a variety of provided DAO map files.
- **Statistics:** Reports path cost, nodes expanded, and running time.
- **Visualization:** Optionally generates images (e.g., `nodes_expanded.png`, `running_time.png`) to visualize search performance.

## Assumptions

- Map files are in the DAO format and located in the `dao-map/` directory.
- The start and goal positions are either hardcoded or provided via command-line arguments.
- The environment is a 2D grid with possible obstacles and varying movement costs.
- Python 3.8+ is required, and dependencies are listed in `requirements.txt`.


## How to Run

1. **Install Dependencies**

   Navigate to the `starter/` directory and install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Program**
    Execute the main script, specifying the map file and (optionally) start/goal positions:
    ```sh
    python main.py --map dao-map/arena.map --start 1,1 --goal 10,10 --algorithm astar
    ```
If arguments are omitted, the program may use default values.

3. **View Results**
- The program prints the path, cost, and statistics to the console.
- If visualization is enabled, check the generated images in the output directory.