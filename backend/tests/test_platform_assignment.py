import numpy as np
import pytest
from train_env import TrainDispatchEnv

def test_early_platform_reservation():
    env = TrainDispatchEnv()
    env.reset()
    
    # Create a simple mock station with 2 platforms
    st_name = 'TEST_STATION'
    switch_in = 1000
    switch_out = 1001
    plat1 = 1002
    plat2 = 1003
    
    env.station_nodes = {
        st_name: {
            'platforms': [plat1, plat2],
            'km': 100.0
        }
    }
    
    env.track_map = {
        switch_in: {'type': 'SWITCH', 'next': [switch_out, plat1, plat2]},
        switch_out: {'type': 'MAIN', 'prev': [switch_in]},
        plat1: {'type': 'PLATFORM', 'station': st_name, 'platform_index': 0, 'capacity': 1},
        plat2: {'type': 'PLATFORM', 'station': st_name, 'platform_index': 1, 'capacity': 1},
    }
    
    # Add stopping train
    env.trains = [
        {'id': 'T1', 'direction': 'DOWN', 'position': switch_in, 'speed': 0, 'target_speed': 0, 'max_speed': 100, 'priority': 5, 'finished': False, 'accel_rate': 10, 'decel_rate': 10, 'visited_nodes': set()},
        {'id': 'T2', 'direction': 'DOWN', 'position': switch_in, 'speed': 0, 'target_speed': 0, 'max_speed': 100, 'priority': 5, 'finished': False, 'accel_rate': 10, 'decel_rate': 10, 'visited_nodes': set()}
    ]
    env._movement_acc = [0.0, 0.0]
    env._train_speeds = [0.0, 0.0]
    env.schedule = {
        'T1': {'stops': [st_name], 'start_time': 0, 'deadline': 1000},
        'T2': {'stops': [], 'start_time': 0, 'deadline': 1000}
    }
    
    # Step 1: Just let them sit at the switch_in and evaluate. 
    env.step(np.array([2, 1]))
    
    # T1 has a scheduled stop, so it should have proactively reserved a platform (plat1 or plat2)
    assert env.trains[0]['committed_next_node'] in [plat1, plat2]
    
    # T2 does not have a scheduled stop, so it defaults to switch_out
    assert env.trains[1]['committed_next_node'] == switch_out


def test_mid_platform_selection_odd_count():
    env = TrainDispatchEnv()
    env.reset()
    
    st_name = 'ODD_STATION'
    switch_in = 2000
    switch_out = 2001
    plats = [2002, 2003, 2004, 2005, 2006] # 5 platforms
    
    env.station_nodes = {
        st_name: {
            'platforms': plats,
            'km': 100.0
        }
    }
    
    env.track_map = {
        switch_in: {'type': 'SWITCH', 'next': [switch_out] + plats, 'prev': [switch_out] + plats},
        switch_out: {'type': 'MAIN', 'prev': [switch_in], 'next': [switch_in]}
    }
    for i, p in enumerate(plats):
        env.track_map[p] = {'type': 'PLATFORM', 'station': st_name, 'platform_index': i, 'capacity': 1}
        
    env.trains = [
        {'id': f'UP{i}', 'direction': 'UP', 'position': switch_in, 'speed': 0, 'target_speed': 0, 'max_speed': 100, 'priority': 5, 'finished': False, 'accel_rate': 10, 'decel_rate': 10, 'visited_nodes': set()}
        for i in range(3)
    ] + [
        {'id': f'DN{i}', 'direction': 'DOWN', 'position': switch_in, 'speed': 0, 'target_speed': 0, 'max_speed': 100, 'priority': 5, 'finished': False, 'accel_rate': 10, 'decel_rate': 10, 'visited_nodes': set()}
        for i in range(3)
    ]
    env._movement_acc = [0.0] * 6
    env._train_speeds = [0.0] * 6
    env.schedule = {t['id']: {'stops': [st_name], 'start_time': 0, 'deadline': 1000} for t in env.trains}
    
    env.step(np.array([2]*6))
    
    res = [t.get('reserved_platform') for t in env.trains]
    assert len(set([r for r in res if r is not None])) == sum(1 for r in res if r is not None), "No two trains should reserve the same platform"
    
    up_res = res[0:3]
    assert up_res[0] == plats[2]
    assert up_res[1] == plats[1]
    assert up_res[2] == plats[0]
    
    dn_res = res[3:5] 
    assert dn_res[0] == plats[3]
    assert dn_res[1] == plats[4]
    assert res[5] is None
