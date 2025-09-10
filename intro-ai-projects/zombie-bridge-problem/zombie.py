from heapq import heappush, heappop
from math import *

class Fringe(object):
    def __init__(self):
        self.heap = []

    def length(self):
        return len(self.heap)

    def add(self, path, priority):
        """Add `path` to the fringe with `priority`"""
        # Push a ``(priority, item)`` tuple onto the heap so that `heappush`
        # and `heappop` will order them properly
        heappush(self.heap, (priority, path))

    def remove(self):
        """Remove and return the earliest-priority path from the fringe"""
        priority,path = heappop(self.heap)
        return path
"""
    The underscore indicates the side the flashlight is on
    _UGPPr| = 1 
    _PPr|UG = 2 
    PrP|UG_ = 3 
    _UG|PrP = 4 
    UG|PrP_ = 5 
    _UP|GPr = 6 
    UP|GPr_ = 7
    _GPr|UP = 8 
    GPr|UP_ = 9 
    _UPr|GP = 10 
    UPr|GP_ = 11 
    _GP|UPr = 12 
    GP|UPr_ = 13 
    _UGP|Pr = 14
    UGP|Pr_ = 15 
    Pr|UGP_ = 16 
    _UGPr|P = 17 
    UGPr|P_ = 18 
    P|UGPr_ = 19 
    _UPPr|G = 20
    UPPr|G_ = 21 
    G|UPPr_ = 22
    _GPPr|U = 23 
    GPPr|U_ = 24 
    U|GPPr_ = 25 
    |UGPPr_ = 26
"""
# Dictionary that contains all the neighbours for each state
neighbours = {1:[3, 21, 18, 15, 24, 9, 13, 11, 7, 5],
    2:[26, 16, 25],
    3:[1, 20, 23],
    4:[26, 22, 25],
    5:[1, 17, 14],
    6:[26, 19, 25],
    7:[1, 14, 20],
    8:[26, 22, 16],
    9:[1, 17, 24],
    10:[26, 25, 16],
    11:[1, 17, 20],
    12:[26, 19, 22],
    13:[1, 14, 23],
    14:[5, 7, 13, 25, 22, 19],
    15:[1],
    16:[10, 8, 2, 23, 20, 17],
    17:[9, 11, 5, 16, 25, 22],
    18:[1],
    19:[6, 2, 12, 14, 23, 20],
    20:[3, 11, 7, 25, 19, 16],
    21:[1],
    22:[4, 12, 8, 23, 17, 14],
    23:[3, 9, 13, 22, 16, 19],
    24:[1],
    25:[4, 6, 10, 14, 17, 20]}

# Dictionary that contains all the costs for every arc
costs = {(1, 24):1, (1, 21):2, (1, 18):5, (1, 15):10, (1, 3):2, (1, 9):5, 
    (1, 13):10, (1, 11):5, (1, 7):10, (1, 5):10, (2, 26):10, (2,16):5,(2, 25):10,
    (3, 1):2, (3, 20):1, (3, 23):2,(4, 26):2, (4, 22): 1,(4, 25):2, (5, 1):10, (5,17):10, 
    (5, 14):5, (6, 26):5, (6, 19):1, (6, 25):5, (7, 1):10, (7, 14):5, (7,20):10,
    (8, 26):10, (8, 22):10, (8, 16):2, (9, 1):10, (9, 17): 1, (9, 24):5, (10, 26):10, (10, 25): 10, 
    (10, 16):1, (11, 1):5, (11, 17):2, (11,20):5, (12, 26):5, (12,19):2, (12,22):5, (13,1):10, (13,14):1, 
    (13,23):10, (14, 5):5, (14,7):2, (14,13):1, (14,25):5, (14,22):5, (14,19):2, (15,1):10, (16, 10):1, 
    (16,8):2, (16,2):5, (16,23):5, (16,20):5, (16,17):2, (17,9):1, (17,11):2, (17,5):10, (17,16):2, 
    (17,25):10, (17,22):10, (18, 1):5, (19, 6):1, (19,2):10, (19,12):2, (19,14):2, (19,23):10, (19,20):10,
    (20,3):1, (20,11):5, (20,7):10, (20,25):10, (20,19):10, (20,16):5, (21, 1):2, (22, 4):1, (22, 12):5, (22,8):10, 
    (22,23):10, (22,16):10, (22,14):5, (23,3):2, (23,9):5, (23,13):10, (23, 22):10, (23, 16):5, (23,19):10,
    (24,1):1, (25, 4):2, (25,6):5, (25, 10):10, (25,14):5, (25,17):10, (25,20):10}

# Heuristic function that gives a list of the time it takes to cross as specified in part 2(d)
heuristic_mapping = {1:[1, 2, 5, 10], 2:[0, 0, 5, 10], 3:[0, 0, 5, 10], 4:[1, 2, 0, 0],
    5:[1, 2, 0, 0], 6:[1, 0, 5, 0], 7:[1, 0, 5, 0], 8:[0, 2, 0, 10], 9:[0, 2, 0, 10],
    10:[1, 0, 0, 10], 11:[1, 0, 0, 10], 12:[1, 0, 5, 0], 13:[1, 0, 5, 0], 14:[1, 2, 5, 0], 15:[1, 2, 5, 0],
    16:[0, 0, 0, 10], 17:[1, 2, 0, 10], 18:[1, 2, 0, 10], 19:[0, 0, 5, 0], 20:[1, 0, 5, 10], 21:[1, 0, 5, 10],
    22:[0, 2, 0, 0], 23:[0, 2, 5, 10], 24:[0, 2, 5, 10], 25:[1, 0, 0, 0], 26:[0, 0, 0, 0]}

class Zombie_problem(object):
    def __init__(self):
        pass
    
    def start_node(self):
        """returns start node"""
        return 1 # _UGPPr|
    
    def is_goal(self, node):
        """is True if `node` is a goal"""
        if node == 26:   #|UGPPr_
            return True
        else: 
            return False

    def neighbors(self, node):
        """returns a list of the arcs for the neighbors of `node`"""
        neighbors_list = neighbours.get(node) # Get the neighbors
        tuple_list = []
        # Iterate through all the neighbours and append them to the list
        for i in neighbors_list:
            x = (node, i)
            tuple_list.append(x)
        return tuple_list

    def arc_cost(self, arc):
        """Returns the cost of `arc`"""
        return costs.get((arc))

    def cost(self, path):
        """Returns the cost of `path`"""
        return sum( self.arc_cost(arc) for arc in path )

    def heuristic(self, node):
        """Returns the heuristic value of `node`"""
        return max(heuristic_mapping.get(node))

    def search(self):
        """Return a solution path"""
        frontier = Fringe() # Initalize the frontier
        # First add all the neighbours of the starting node to the frontier
        neigh = self.neighbors(1) # Get the neighbors of the starting node
        # Add the arc cost and the heuristic together and add it to the frontier
        # along with the arc itself
        for x in neigh:
            arc = (x)
            prio = self.arc_cost(arc) 
            prio += max(heuristic_mapping.get(1))
            frontier.add([arc], prio) # Add to frontier as a tuple
        solution_path = [] # Stores the solution path
        # Continually remove the f-minimizing path from the frontier until
        # the length of the frontier is 0 or the end goal is reached
        while frontier.length() != 0:
            fmin = frontier.remove() # Remove and store the f-minimizing path  
            solution_path.append(fmin[0]) # Append the path to the solution path 
            # If a goal node is reached return the solution path
            if fmin[0][1] == 26:
                solution_path.pop(0) # Get rid of extraneous 
                return solution_path
            neigh = self.neighbors(fmin[0][1]) # Get the neighbors of the current f-min path
            # Iterate through the neighbors of the f-min path
            for x in neigh:
                arc = (x)
                new_arc = arc[::-1] # Get the reverse of the current arc
                # If the current arc and its reverse are in the solution path
                # then we have backtracked, these paths should be removed (cycle pruning)
                if arc in solution_path and new_arc in solution_path:
                    solution_path.remove(new_arc)
                else:
                    prio = self.arc_cost(arc) # Calculate the arc cost of the neighbor
                    prio += max(heuristic_mapping.get(arc[1])) # Add the heuristic value
                    frontier.add([arc], prio) # Add the arc to the solution path with its 
                    # arc cost + heuristic value

def unit_tests():
    """
    Some trivial tests to check that the implementation even runs.
    Feel free to add additional tests.
    """
    print("testing...")
    p = Zombie_problem()
    assert p.start_node() is not None
    assert not p.is_goal(p.start_node())
    assert p.heuristic(p.start_node()) >= 0

    ns = p.neighbors(p.start_node())
    assert len(ns) > 0

    soln = p.search()
    assert p.cost(soln) > 0
    print("tests ok")

def main():
    unit_tests()
    p = Zombie_problem()
    soln = p.search()
    if soln:
        print("Solution found (cost=%s)\n%s" % (p.cost(soln), soln))
    else:
        raise RuntimeError("Empty solution")

if __name__ == '__main__':
    main()
