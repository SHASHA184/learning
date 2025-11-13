from collections import deque
from typing import List, Optional, Tuple

import requests

BASE_URL = "https://c20240327a.challenges.weblab.technology"


def fetch_neighbors(node: str) -> List[str]:
    url = f"{BASE_URL}/{node}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text.strip().split()


def find_path_bfs(start: str, target: str) -> Optional[Tuple[List[str], int]]:
    """BFS guarantees shortest path in unweighted graph."""
    queue = deque([(start, [start])])
    visited = {start}
    request_count = 0

    while queue:
        current_node, path = queue.popleft()
        neighbors = fetch_neighbors(current_node)
        request_count += 1
        print(f"Visited /{current_node}: neighbors = {neighbors}")

        if target in neighbors:
            final_path = path + [target]
            print("\n")
            print(f"Target '{target}' found!")
            print(f"Path: {' -> '.join(final_path)}")
            print(f"HTTP requests made: {request_count}")
            return final_path, request_count

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    print(f"\nNo path found from '{start}' to '{target}'")
    return None


def main():
    print("Finding the shortest path from 'a' to 'x'")
    print("\n")
    result = find_path_bfs("a", "x")

    if result:
        path, request_count = result
        print("\n")
        print(f"Shortest path: {' -> '.join(path)}")
        print(f"Path length: {len(path) - 1} hops")
        print(f"Total HTTP requests: {request_count}")


if __name__ == "__main__":
    main()
