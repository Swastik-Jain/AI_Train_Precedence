import pytest
from ai.map_generator import GhatTokenSystem

@pytest.fixture
def synthetic_map():
    # 0(non-token) -> 1(token, KSR gate) -> 2(token) -> 3(token, IGP gate) -> 4(non-token)
    # prev/next both populated both directions, matching real generation pattern.
    return {
        0: {'next': [1], 'prev': [], 'token_block': False},
        1: {'next': [2], 'prev': [0], 'token_block': True},
        2: {'next': [3], 'prev': [1], 'token_block': True},
        3: {'next': [4], 'prev': [2], 'token_block': True},
        4: {'next': [], 'prev': [3], 'token_block': False},
    }

@pytest.fixture
def token_system():
    return GhatTokenSystem([1, 2, 3])

def test_empty_occupied(synthetic_map, token_system):
    # 1. Empty occupied_by_node -> both sides return []
    assert token_system.compute_queue(synthetic_map, {}, 'KSR') == []
    assert token_system.compute_queue(synthetic_map, {}, 'IGP') == []

def test_one_down_train(synthetic_map, token_system):
    # 2. One DOWN train on node 0 -> KSR returns ['T1']
    occupied = {0: {'train_id': 'T1', 'direction': 'DOWN'}}
    assert token_system.compute_queue(synthetic_map, occupied, 'KSR') == ['T1']
    assert token_system.compute_queue(synthetic_map, occupied, 'IGP') == []

def test_multiple_down_trains(token_system):
    # 3. DOWN trains on nodes 0 and the next node back -> both returned, nearest-to-ghat first.
    # Extend synthetic map
    extended_map = {
        -1: {'next': [0], 'prev': [], 'token_block': False},
        0: {'next': [1], 'prev': [-1], 'token_block': False},
        1: {'next': [2], 'prev': [0], 'token_block': True},
        2: {'next': [3], 'prev': [1], 'token_block': True},
        3: {'next': [4], 'prev': [2], 'token_block': True},
        4: {'next': [], 'prev': [3], 'token_block': False},
    }
    occupied = {
        0: {'train_id': 'T1', 'direction': 'DOWN'},
        -1: {'train_id': 'T2', 'direction': 'DOWN'}
    }
    assert token_system.compute_queue(extended_map, occupied, 'KSR') == ['T1', 'T2']

def test_gap_case(token_system):
    # 4. Gap case: DOWN train on node one step further back than node 0, but node 0 itself empty -> returns []
    extended_map = {
        -1: {'next': [0], 'prev': [], 'token_block': False},
        0: {'next': [1], 'prev': [-1], 'token_block': False},
        1: {'next': [2], 'prev': [0], 'token_block': True},
        2: {'next': [3], 'prev': [1], 'token_block': True},
        3: {'next': [4], 'prev': [2], 'token_block': True},
        4: {'next': [], 'prev': [3], 'token_block': False},
    }
    occupied = {
        -1: {'train_id': 'T2', 'direction': 'DOWN'}
    }
    assert token_system.compute_queue(extended_map, occupied, 'KSR') == []

def test_wrong_direction(synthetic_map, token_system):
    # 5. Wrong-direction train on node 0 (UP train on a KSR-side node) -> []
    occupied = {0: {'train_id': 'T1', 'direction': 'UP'}}
    assert token_system.compute_queue(synthetic_map, occupied, 'KSR') == []

def test_train_inside_token_block(synthetic_map, token_system):
    # 6. A train inside the token block itself is never returned by either side's compute_queue
    occupied = {
        1: {'train_id': 'T1', 'direction': 'DOWN'},
        0: {'train_id': 'T2', 'direction': 'DOWN'}
    }
    assert token_system.compute_queue(synthetic_map, occupied, 'KSR') == ['T2']

def test_max_hops_cap(token_system):
    # 7. max_hops cap: occupy max_hops + 2 consecutive nodes on a longer synthetic chain -> returned list length is exactly max_hops
    # KSR max_hops is 8
    long_map = {}
    for i in range(-10, 1):
        long_map[i] = {'next': [i+1], 'prev': [i-1], 'token_block': False}
    long_map[0]['next'] = [1]
    long_map[1] = {'next': [2], 'prev': [0], 'token_block': True}
    long_map[2] = {'next': [3], 'prev': [1], 'token_block': True}
    
    occupied = {}
    for i in range(0, -10, -1): # 0 to -9 (10 nodes)
        occupied[i] = {'train_id': f'T{abs(i)}', 'direction': 'DOWN'}
        
    queue = token_system.compute_queue(long_map, occupied, 'KSR')
    assert len(queue) == 8
    assert queue == [f'T{i}' for i in range(8)]
