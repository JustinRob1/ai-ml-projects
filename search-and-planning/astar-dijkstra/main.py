import time
from search.algorithms import State
from search.map import Map
import getopt
import sys

import heapq

def main():
    """
    Function for testing your A* and Dijkstra's implementation. 
    Run it with a -help option to see the options available. 
    """
    optlist, _ = getopt.getopt(sys.argv[1:], 'h:m:r:', ['testinstances', 'plots', 'help'])

    plots = False
    for o, a in optlist:
        if o in ("-help"):
            print("Examples of Usage:")
            print("Solve set of test instances and generate plots: main.py --plots")
            exit()
        elif o in ("--plots"):
            plots = True

    test_instances = "test-instances/testinstances.txt"
    
    # Dijkstra's algorithm and A* should receive the following map object as input
    gridded_map = Map("dao-map/brc000d.map")
    
    nodes_expanded_dijkstra = []  
    nodes_expanded_astar = []

    time_dijkstra = []  
    time_astar = []

    start_states = []
    goal_states = []
    solution_costs = []
       
    file = open(test_instances, "r")
    for instance_string in file:
        list_instance = instance_string.split(",")
        start_states.append(State(int(list_instance[0]), int(list_instance[1])))
        goal_states.append(State(int(list_instance[2]), int(list_instance[3])))
        
        solution_costs.append(float(list_instance[4]))
    file.close()
        
    for i in range(0, len(start_states)):    
        start = start_states[i]
        goal = goal_states[i]
    
        time_start = time.time()
        cost, expanded_diskstra = dijkstra(start, goal, gridded_map)
        time_end = time.time()
        nodes_expanded_dijkstra.append(expanded_diskstra)
        time_dijkstra.append(time_end - time_start)

        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by Dijkstra and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()    

        start = start_states[i]
        goal = goal_states[i]
    
        time_start = time.time()
        cost, expanded_astar = astar(start, goal, gridded_map)
        time_end = time.time()

        nodes_expanded_astar.append(expanded_astar)
        time_astar.append(time_end - time_start)

        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by A* and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()

    if plots:
        from search.plot_results import PlotResults
        plotter = PlotResults()
        plotter.plot_results(nodes_expanded_astar, nodes_expanded_dijkstra, "Nodes Expanded (A*)", "Nodes Expanded (Dijkstra)", "nodes_expanded")
        plotter.plot_results(time_astar, time_dijkstra, "Running Time (A*)", "Running Time (Dijkstra)", "running_time")

def dijkstra(start, goal, gridded_map):
    OPEN = [start]
    CLOSED = {}
    expansions = 0
    
    while OPEN:
        n = heapq.heappop(OPEN)

        if n == goal:
            return n.get_cost(), expansions
        
        expansions += 1

        for s in gridded_map.successors(n):
            s.set_cost(s.get_g())
            if s.state_hash() not in CLOSED or s.get_cost() < CLOSED[s.state_hash()].get_cost():
                CLOSED[s.state_hash()] = s
                heapq.heappush(OPEN, s)
    return -1, expansions

def astar(start, goal, gridded_map):
    OPEN = [start]
    CLOSED = {}
    expansions = 0
    
    while OPEN:
        n = heapq.heappop(OPEN)

        if n.state_hash() in CLOSED and n.get_cost() > CLOSED[n.state_hash()].get_cost():
            continue

        if n == goal:
            return n.get_cost(), expansions
        
        expansions += 1
        
        for s in gridded_map.successors(n):
            s.set_cost(s.get_g() + octile_distance(s, goal))
            if s.state_hash() not in CLOSED or s.get_cost() < CLOSED[s.state_hash()].get_cost():
                CLOSED[s.state_hash()] = s
                heapq.heappush(OPEN, s)
    return -1, expansions
    
def octile_distance(state, goal):
    dx = abs(state.get_x() - goal.get_x())
    dy = abs(state.get_y() - goal.get_y())
    return 1.5 * min(dx, dy) + abs(dx - dy)

if __name__ == "__main__":
    main()