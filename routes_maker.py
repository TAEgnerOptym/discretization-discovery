from collections import defaultdict

# Input arcs (u → v)
arcs = [
    (0, 2), (1, 0), (2, 4), (3, 1), (4, 26), (5, 3), (6, 7), (7, 9), (8, 5),
    (9, 10), (10, 8), (11, 26), (12, 16), (13, 11), (14, 15), (15, 13),
    (16, 17), (17, 18), (18, 14), (19, 23), (20, 26), (21, 20), (22, 21),
    (23, 24), (24, 22), (25, 6), (25, 12), (25, 19)
]

# Step 1: Build successor map
succ_map = defaultdict(list)
for u, v in arcs:
    succ_map[u].append(v)

# Step 2: DFS from 25 to 26
def dfs(node, path, all_paths):
    if node == 26:
        all_paths.append(path[:])
        return
    for neighbor in succ_map.get(node, []):
        path.append(neighbor)
        dfs(neighbor, path, all_paths)
        path.pop()

# Step 3: Collect and print all paths
all_paths = []
dfs(25, [25], all_paths)

for route in all_paths:
    print(" → ".join(map(str, route)))
