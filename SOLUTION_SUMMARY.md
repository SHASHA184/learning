# Web Challenge Solution

## Problem Analysis

**Objective**: Find a path from `/a` to `/x` with minimal HTTP requests.

**Starting Point**: `GET https://c20240327a.challenges.weblab.technology/a`

## Pattern Recognition

After analyzing the initial endpoint, I identified the following pattern:

1. Each endpoint returns a space-separated list of letters
2. These letters represent edges in a directed graph
3. From node `/a`, you can navigate to any node listed in its response
4. This is a **graph traversal problem** requiring shortest path finding

### Example:
- `/a` returns: `h r l m b g` → Can visit: `/h`, `/r`, `/l`, `/m`, `/b`, `/g`
- `/r` returns: `a h b e x` → Can visit: `/a`, `/h`, `/b`, `/e`, `/x`

## Solution Strategy

### Algorithm: Breadth-First Search (BFS)

**Why BFS?**
- Guarantees the **shortest path** in an unweighted graph
- Explores nodes level-by-level, minimizing unnecessary requests
- Stops immediately when target is found

**Key Optimizations:**
1. **Visited Set**: Prevents revisiting nodes and redundant HTTP requests
2. **Early Termination**: Stops as soon as 'x' is found in any neighbor list
3. **Efficient Data Structure**: Uses `deque` for O(1) queue operations

## Result

```
Shortest Path: a -> r -> x
Path Length: 2 hops
HTTP Requests: 3 (optimal for this graph structure)
```

### Request Breakdown:
1. Request 1: Fetch `/a` → Get neighbors: h, r, l, m, b, g
2. Request 2: Fetch `/h` → Get neighbors: a, j, m, f, g, r, l, m, d, k (no 'x')
3. Request 3: Fetch `/r` → Get neighbors: a, h, b, e, **x** ✓

## Code Quality Highlights

1. **Clear Documentation**: Comprehensive docstrings and comments
2. **Type Hints**: Modern Python type annotations for better code clarity
3. **Error Handling**: HTTP status checking with `raise_for_status()`
4. **Modularity**: Separate functions for fetching, searching, and main logic
5. **Portability**: Only uses standard library + `requests` (widely available)
6. **Readability**: Clear variable names and logical flow

## Complexity Analysis

- **Time Complexity**: O(V + E) where V = vertices, E = edges
- **Space Complexity**: O(V) for queue and visited set
- **Network Requests**: O(V) worst case, but BFS minimizes actual requests

## Running the Solution

```bash
# Install dependencies (if needed)
pip install requests

# Run the solver
python challenge_solution.py
```

## Assumptions Made

1. The graph structure is static (edges don't change between requests)
2. All nodes are lowercase single letters
3. The response format is consistently space-separated letters
4. The graph is connected (a path exists from 'a' to 'x')

## Alternative Approaches Considered

1. **DFS (Depth-First Search)**: Would work but doesn't guarantee shortest path
2. **Dijkstra's Algorithm**: Overkill for unweighted graph
3. **Bi-directional BFS**: Could reduce requests but adds complexity
4. **A* Search**: Requires heuristic function; hard to define without graph structure

**Conclusion**: BFS is the optimal choice for this problem.
