import json
from ai.map_generator import generate_realistic_section

def get_network_topology():
    track_map, loop_sections, destination_id, station_nodes, token_blocks = generate_realistic_section()
    
    nodes = []
    edges = []
    
    # We will do a basic BFS/Layout heuristic to assign X and Y coordinates.
    # Root is node 0.
    # X increases by 100 for each step forward
    # Y branches off by +/- 50 for loops
    
    layout = {}
    visited = set()
    queue = [(0, 100, 250)] # (node_id, x, y)
    
    while queue:
        node, x, y = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        
        info = track_map.get(node)
        if not info:
            continue
            
        layout[node] = {
            "id": str(node), 
            "x": x, 
            "y": y, 
            "type": info.get('type', 'NODE'),
            "km": info.get('km', 0),
            "stId": info.get('station') or info.get('label'),
            "capacity": info.get('capacity', 2),
            "platform_index": info.get('platform_index'),
            "loop_index": info.get('loop_index'),
            "token_block": info.get('token_block', False)
        }
        
        next_nodes = info.get('next', [])
        if len(next_nodes) == 1:
            queue.append((next_nodes[0], x + 150, y))
        elif len(next_nodes) > 1:
            # multiple branches, spread them out in Y
            y_offset = -24 * ((len(next_nodes) - 1) / 2.0)
            for nxt in next_nodes:
                queue.append((nxt, x + 180, y + y_offset))
                y_offset += 24

    # Build node list
    for node_id, data in layout.items():
        nodes.append(data)
        
    # Build edges
    for node_id, info in track_map.items():
        if node_id not in layout:
            continue
        next_nodes = info.get('next', [])
        for nxt in next_nodes:
            if nxt in layout:
                edge_id = f"edge-{node_id}-{nxt}"
                edges.append({
                    "id": edge_id,
                    "source": str(node_id),
                    "target": str(nxt),
                    "length": 1, # logical length
                    "max_speed": info.get("speed", 100),
                    "capacity": info.get("capacity", 1),
                    "type": "track"
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "raw": {
            "track_map": track_map,
            "loop_sections": loop_sections,
            "destination_id": destination_id
        }
    }

if __name__ == "__main__":
    t = get_network_topology()
    print(json.dumps(t, indent=2))
