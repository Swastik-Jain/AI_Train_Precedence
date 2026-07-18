"""
map_generator.py — CSMT → Manmad Corridor
Real topology based on Central Railway Bhusawal-Kalyan section.

Corridor: CSMT(0km) → Dadar(9km) → Kalyan(54km) → Kasara(121km)
          → [GHAT TOKEN BLOCK 15km, gradient 1:37] →
          Igatpuri(136km) → Devlali(182km) → Nashik Road(187km) → Manmad(261km)

Node ID scheme:
  Main line blocks : 1–999   (sequential, ordered by km from CSMT)
  Platform nodes   : 1000+   (grouped by station, inherit station km)
  Token block      : tagged with 'token_block': True on Kasara→Igatpuri blocks

Track type per segment:
  CSMT  → Kalyan  : QUADRUPLE (capacity 4, speed 110–130)
  Kalyan→ Kasara  : DOUBLE    (capacity 2, speed 100)
  Kasara→ Igatpuri: GHAT      (capacity 1 mid-line token, speed 50 uphill / 60 downhill)
  Igatpuri→ Manmad: DOUBLE    (capacity 2, speed 110–130)
"""

from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# SECTION LENGTH — used for observation normalization
# ─────────────────────────────────────────────────────────────────────────────
SECTION_LENGTH_KM = 261  # CSMT → Manmad

# ─────────────────────────────────────────────────────────────────────────────
# STATION DEFINITIONS — real km, platforms, loops
# ─────────────────────────────────────────────────────────────────────────────
STATIONS = {
    'CSMT': {
        'km': 0,
        'code': 'CSMT',
        'platforms': 4,
        'loops': 2,
        'type': 'TERMINUS',
        'speed_limit': 30,
        'is_banker_point': False,
    },
    'DADAR': {
        'km': 9,
        'code': 'DR',
        'platforms': 3,
        'loops': 1,
        'type': 'MAJOR_JUNCTION',
        'speed_limit': 50,
        'is_banker_point': False,
    },
    'KALYAN': {
        'km': 54,
        'code': 'KYN',
        'platforms': 7,
        'loops': 3,
        'type': 'MAJOR_JUNCTION',
        'speed_limit': 50,
        'is_banker_point': False,
        # Suburban ends here. All trains beyond Kalyan stop at every station.
        'suburban_terminus': True,
    },
    'AMBERNATH': {
        'km': 63,
        'code': 'AMBN',
        'platforms': 0,   # crossing loop only — no passenger platforms
        'loops': 2,
        'type': 'CROSSING_LOOP',
        'speed_limit': 75,
        'is_banker_point': False,
    },
    'TITWALA': {
        'km': 80,
        'code': 'TTWL',
        'platforms': 0,   # crossing loop only — no passenger platforms
        'loops': 2,
        'type': 'CROSSING_LOOP',
        'speed_limit': 75,
        'is_banker_point': False,
    },
    'ATGAON': {
        'km': 98,
        'code': 'ATGN',
        'platforms': 0,   # crossing loop only — no passenger platforms
        'loops': 2,
        'type': 'CROSSING_LOOP',
        'speed_limit': 75,
        'is_banker_point': False,
    },
    'KASARA': {
        'km': 121,
        'code': 'KSRA',
        'platforms': 4,
        'loops': 3,
        'type': 'STATION',
        'speed_limit': 30,
        'is_banker_point': True,   # ← banker locos attach here (UP trains)
        'ghat_start': True,
    },
    'IGATPURI': {
        'km': 136,
        'code': 'IGP',
        'platforms': 6,
        'loops': 4,
        'type': 'MAJOR_STATION',
        'speed_limit': 30,
        'is_banker_point': True,   # ← banker locos detach here (UP trains), attach (DOWN trains)
        'ghat_end': True,
    },
    'DEVLALI': {
        'km': 182,
        'code': 'DVL',
        'platforms': 3,
        'loops': 2,
        'type': 'STATION',
        'speed_limit': 75,
        'is_banker_point': False,
    },
    'NASHIK': {
        'km': 187,
        'code': 'NK',
        'platforms': 6,
        'loops': 3,
        'type': 'MAJOR_JUNCTION',
        'speed_limit': 50,
        'is_banker_point': False,
    },
    'MANMAD': {
        'km': 261,
        'code': 'MMR',
        'platforms': 4,
        'loops': 2,
        'type': 'TERMINUS',
        'speed_limit': 30,
        'is_banker_point': False,
    },
}

# Station order — fixed, CSMT→Manmad (DOWN direction)
STATION_ORDER = ['CSMT', 'DADAR', 'KALYAN', 'AMBERNATH', 'TITWALA', 'ATGAON', 'KASARA', 'IGATPURI', 'DEVLALI', 'NASHIK', 'MANMAD']

# ─────────────────────────────────────────────────────────────────────────────
# INTER-STATION BLOCK SPECS
# ─────────────────────────────────────────────────────────────────────────────
# Each entry defines the blocks BETWEEN two adjacent stations.
# 'blocks'      : number of inter-station blocks (approx 5km each)
# 'capacity'    : trains per block (4=quad, 2=double, 1=ghat single)
# 'speed'       : max speed on these blocks (km/h)
# 'token_block' : True only for Kasara-Igatpuri ghat mid-line
# 'gradient'    : True for ghat section (affects speed profile)

INTER_STATION_SPECS = {
    ('CSMT', 'DADAR'): {
        'blocks': 2,
        'capacity': 4,       # quadruple track
        'speed': 110,
        'token_block': False,
        'gradient': False,
    },
    ('DADAR', 'KALYAN'): {
        'blocks': 9,
        'capacity': 3,       # quadruple track reduced to 3 — suburban congestion
        'speed': 130,
        'token_block': False,
        'gradient': False,
    },
    # KALYAN→KASARA split into 4 sub-segments via crossing stations
    # Each sub-segment has its own blocks proportional to distance
    ('KALYAN', 'AMBERNATH'): {
        'blocks': 2,         # 63-54 = 9km, ~4.5km per block
        'capacity': 2,       # double track
        'speed': 100,
        'token_block': False,
        'gradient': False,
    },
    ('AMBERNATH', 'TITWALA'): {
        'blocks': 4,         # 80-63 = 17km, ~4.25km per block
        'capacity': 2,       # double track
        'speed': 100,
        'token_block': False,
        'gradient': False,
    },
    ('TITWALA', 'ATGAON'): {
        'blocks': 4,         # 98-80 = 18km, ~4.5km per block
        'capacity': 2,       # double track
        'speed': 100,
        'token_block': False,
        'gradient': False,
    },
    ('ATGAON', 'KASARA'): {
        'blocks': 5,         # 121-98 = 23km, ~4.6km per block
        'capacity': 2,       # double track
        'speed': 100,
        'token_block': False,
        'gradient': False,
    },
    ('KASARA', 'IGATPURI'): {
        'blocks': 8,         # 15km ghat, ~1.9km per block — was 3, now 8
        'capacity': 1,       # ← THE BOTTLENECK: mid-line token block
        'speed': 50,         # speed restricted, 1:37 gradient
        'token_block': True, # bidirectional token working
        'gradient': True,
    },
    ('IGATPURI', 'DEVLALI'): {
        'blocks': 9,
        'capacity': 2,       # double track beyond ghat
        'speed': 110,
        'token_block': False,
        'gradient': False,
    },
    ('DEVLALI', 'NASHIK'): {
        'blocks': 3,         # was 1 — 5km at ~1.7km per block, more realistic
        'capacity': 2,
        'speed': 75,
        'token_block': False,
        'gradient': False,
    },
    ('NASHIK', 'MANMAD'): {
        'blocks': 15,
        'capacity': 2,
        'speed': 130,
        'token_block': False,
        'gradient': False,
    },
}


# ────────────────────────────────────────────────────────────────────────────────
# MID-SECTION CROSSING LOOPS
# These are standalone siding tracks between stations — not at a station.
# Each entry: km position → number of loop tracks at that location.
# The generator attaches them to the nearest main-line block at that km.
# Wiring: main_block['next'] = [next_main_block, crossing_loop_node]
#         crossing_loop_node['next'] = [next_main_block]   ← rejoins after
# This means a train in the loop still progresses — it re-enters main line
# at the NEXT block after the loop, just like a real passing loop.
# ────────────────────────────────────────────────────────────────────────────────
MID_SECTION_CROSSING_LOOPS = {
    210.0: {'loops': 2, 'speed': 50, 'label': 'LOOP_NANDGAON'},   # Nashik→Manmad km 210
    235.0: {'loops': 2, 'speed': 50, 'label': 'LOOP_LASALGAON'},  # Nashik→Manmad km 235
}


def generate_realistic_section():
    """
    Build the CSMT–Manmad corridor graph.

    Returns
    -------
    track_map   : dict[node_id → node_data]
    loop_sections : list[node_id]  — all platform/loop nodes
    end_node    : int              — destination node ID (999)
    station_nodes : dict[station_name → dict]  — anchor node IDs per station
    token_blocks  : list[node_id]  — Kasara-Igatpuri ghat block node IDs
    """

    track_map = {}
    loop_sections = []
    token_blocks = []

    # station_nodes maps station name → {
    #   'switch_in': node_id,   ← main line node just before platforms
    #   'platforms': [node_ids],
    #   'loops': [node_ids],
    #   'switch_out': node_id,  ← main line node just after platforms
    #   'km': float
    # }
    station_nodes = {}

    current_node = 0
    platform_node_base = 1000   # platform/loop nodes start at 1000

    # ── Node 0: Origin / spawn point ──────────────────────────────────────
    track_map[0] = {
        'type': 'ORIGIN',
        'speed': 0,
        'capacity': 99,
        'next': [1],
        'km': 0.0,
        'station': None,
        'token_block': False,
        'gradient': False,
    }
    current_node = 1

    for s_idx, station_name in enumerate(STATION_ORDER):
        st = STATIONS[station_name]
        st_km = st['km']
        n_platforms = st['platforms']
        n_loops = st['loops']

        # ── SWITCH IN — first main-line node of this station ──────────────
        switch_in = current_node
        track_map[switch_in] = {
            'type': 'SWITCH',
            'speed': st['speed_limit'],
            'capacity': 4 if st_km < 55 else 2,  # inherit from segment
            'next': [],   # filled after platform nodes are known
            'km': st_km,
            'station': station_name,
            'token_block': False,
            'gradient': False,
            'is_banker_point': st.get('is_banker_point', False),
        }
        current_node += 1

        # ── PLATFORM NODES ─────────────────────────────────────────────────
        platform_ids = []
        for p in range(n_platforms):
            pid = platform_node_base
            platform_node_base += 1
            track_map[pid] = {
                'type': 'PLATFORM',
                'speed': st['speed_limit'],
                'capacity': 1,
                'next': [],   # filled below (→ switch_out)
                'km': st_km,
                'station': station_name,
                'platform_index': p,
                'token_block': False,
                'gradient': False,
            }
            platform_ids.append(pid)
            loop_sections.append(pid)

        # ── LOOP NODES ─────────────────────────────────────────────────────
        loop_ids = []
        for l in range(n_loops):
            lid = platform_node_base
            platform_node_base += 1
            track_map[lid] = {
                'type': 'LOOP',
                'speed': 30,
                'capacity': 1,
                'next': [],   # filled below
                'km': st_km,
                'station': station_name,
                'loop_index': l,
                'token_block': False,
                'gradient': False,
            }
            loop_ids.append(lid)
            loop_sections.append(lid)

        # ── SWITCH OUT — last main-line node of this station ───────────────
        switch_out = current_node
        is_last_station = (s_idx == len(STATION_ORDER) - 1)
        switch_out_next = [999] if is_last_station else [current_node + 1]

        track_map[switch_out] = {
            'type': 'SWITCH',
            'speed': st['speed_limit'],
            'capacity': 4 if st_km < 55 else 2,
            'next': switch_out_next,
            'km': st_km,
            'station': station_name,
            'token_block': False,
            'gradient': False,
        }
        current_node += 1

        # ── Wire platform/loop → switch_out ───────────────────────────────
        for pid in platform_ids + loop_ids:
            track_map[pid]['next'] = [switch_out]

        # ── Wire switch_in → [switch_out (main), platforms, loops] ────────
        # Main path is switch_out directly (trains that don't stop),
        # then platforms (scheduled stops), then loops (diversions/holds)
        track_map[switch_in]['next'] = [switch_out] + platform_ids + loop_ids

        # ── Store station anchor data ──────────────────────────────────────
        station_nodes[station_name] = {
            'switch_in': switch_in,
            'switch_out': switch_out,
            'platforms': platform_ids,
            'loops': loop_ids,
            'km': st_km,
            'is_banker_point': st.get('is_banker_point', False),
            'ghat_start': st.get('ghat_start', False),
            'ghat_end': st.get('ghat_end', False),
        }

        # ── INTER-STATION BLOCKS to next station ──────────────────────────
        if not is_last_station:
            next_station = STATION_ORDER[s_idx + 1]
            seg = INTER_STATION_SPECS[(station_name, next_station)]
            n_blocks = seg['blocks']
            seg_speed = seg['speed']
            seg_cap = seg['capacity']
            is_token = seg['token_block']
            is_grad = seg['gradient']

            # km per block in this segment
            km_start = st_km
            km_end = STATIONS[next_station]['km']
            km_per_block = (km_end - km_start) / n_blocks

            for b in range(n_blocks):
                block_km = km_start + (b + 1) * km_per_block
                block_id = current_node
                is_last_block = (b == n_blocks - 1)
                next_ids = [current_node + 1] if not is_last_block else [
                    station_nodes[next_station]['switch_in']
                    if next_station in station_nodes
                    else current_node + 1
                ]

                track_map[block_id] = {
                    'type': 'GHAT_BLOCK' if is_token else 'MAIN_BLOCK',
                    'speed': seg_speed,
                    'capacity': seg_cap,
                    'next': next_ids,
                    'km': round(block_km, 1),
                    'station': None,
                    'token_block': is_token,
                    'gradient': is_grad,
                }

                if is_token:
                    token_blocks.append(block_id)

                current_node += 1

            # Fix: last inter-station block must point to next station's switch_in
            # (switch_in doesn't exist yet when we write inter-station blocks
            #  for the first stations — fix forward reference here)
            if next_station in station_nodes:
                last_block_id = current_node - 1
                track_map[last_block_id]['next'] = [station_nodes[next_station]['switch_in']]

    # ── Mid-section crossing loops ────────────────────────────────────────────────────────────
    # For each km position in MID_SECTION_CROSSING_LOOPS, find the nearest
    # MAIN_BLOCK node. Wire it so:
    #   main_block['next'] = [original_next_block, loop_node_1, loop_node_2, ...]
    #   loop_node['next']  = [original_next_block]   ← loop rejoins after block
    # This gives trains a DIVERT option at that block: take the loop and
    # rejoin the main line at the same exit point, having yielded the block.

    for loop_km, loop_spec in MID_SECTION_CROSSING_LOOPS.items():
        # Find nearest MAIN_BLOCK node to this km
        nearest_id = None
        nearest_dist = float('inf')
        for nid, nd in track_map.items():
            if nd.get('type') == 'MAIN_BLOCK' and nd.get('station') is None:
                dist = abs(nd['km'] - loop_km)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = nid

        if nearest_id is None:
            continue

        # The main block's current next (must have exactly 1 next to be safe)
        original_next = track_map[nearest_id]['next']
        if not original_next:
            continue
        rejoin_node = original_next[0]

        # Create loop nodes for this crossing loop
        new_loop_ids = []
        for l_idx in range(loop_spec['loops']):
            lid = platform_node_base
            platform_node_base += 1
            track_map[lid] = {
                'type': 'CROSSING_LOOP',
                'speed': loop_spec['speed'],
                'capacity': 1,
                'next': [rejoin_node],   # ← loop rejoins main line after this block
                'km': loop_km,
                'station': None,
                'token_block': False,
                'gradient': False,
                'label': loop_spec['label'],
                'loop_index': l_idx,
            }
            new_loop_ids.append(lid)
            loop_sections.append(lid)

        # Wire the main block: [original_next] + loop_nodes
        track_map[nearest_id]['next'] = original_next + new_loop_ids

    # ── Destination node ──────────────────────────────────────────────────
    track_map[999] = {
        'type': 'DESTINATION',
        'speed': 0,
        'capacity': 99,
        'next': [],
        'km': SECTION_LENGTH_KM,
        'station': 'MANMAD',
        'token_block': False,
        'gradient': False,
    }

    # ── Populate backward links (prev) ────────────────────────────────────
    # Ensure every node has a 'prev' list
    for nid, ndata in track_map.items():
        if 'prev' not in ndata:
            ndata['prev'] = []
    
    # Populate the lists based on forward 'next' links
    for nid, ndata in track_map.items():
        for target_id in ndata.get('next', []):
            if target_id in track_map:
                if 'prev' not in track_map[target_id]:
                    track_map[target_id]['prev'] = []
                if nid not in track_map[target_id]['prev']:
                    track_map[target_id]['prev'].append(nid)

    return track_map, loop_sections, 999, station_nodes, token_blocks


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN BLOCK STATE — Kasara-Igatpuri mid-line
# ─────────────────────────────────────────────────────────────────────────────

class GhatTokenSystem:
    """
    Manages bidirectional token working on the Kasara-Igatpuri ghat section.

    Real operation:
    - UP trains (CSMT→Manmad): hold token while in ghat, release at Igatpuri
    - DOWN trains (Manmad→CSMT): hold token while in ghat, release at Kasara
    - Only one direction may hold the token at any time
    - Banker locos count as token holders

    Usage:
        token = GhatTokenSystem(token_block_ids)
        token.reset()
        can_enter = token.can_enter(train_id, direction)   # before action mask
        token.train_entered(train_id, direction)           # when train moves in
        token.train_exited(train_id)                       # when train clears
    """

    def __init__(self, token_block_ids: list):
        self.token_block_ids = set(token_block_ids)
        self.reset()

    def reset(self):
        # Current direction holding the token: 'UP', 'DOWN', or None
        self.token_direction = None
        # Set of train IDs currently inside the token block
        self.trains_in_block = set()

    def can_enter(self, train_id: str, direction: str) -> bool:
        """
        Returns True if train can enter the ghat token block.
        - If block is free: always yes
        - If block held by same direction: yes (convoy allowed)
        - If block held by opposing direction: NO
        """
        if not self.trains_in_block:
            return True
        if self.token_direction == direction:
            return True  # same direction convoy
        return False     # opposing direction blocked

    def train_entered(self, train_id: str, direction: str):
        """Call when a train successfully moves into a token block node."""
        self.trains_in_block.add(train_id)
        self.token_direction = direction

    def train_exited(self, train_id: str):
        """Call when a train moves out of the last token block node."""
        self.trains_in_block.discard(train_id)
        if not self.trains_in_block:
            self.token_direction = None   # block free, direction released

    def compute_queue(self, track_map: dict, occupied_by_node: dict, side: str) -> list[str]:
        if not self.token_block_ids:
            return []
        
        if side not in ('KSR', 'IGP'):
            raise ValueError("side must be 'KSR' or 'IGP'")
            
        entry_start = None
        if side == 'KSR':
            candidates = []
            for tid in self.token_block_ids:
                prev_nodes = track_map.get(tid, {}).get('prev', [])
                if any(p not in self.token_block_ids for p in prev_nodes):
                    candidates.append(tid)
            if candidates:
                gate_node = min(candidates)
                prev_nodes = track_map.get(gate_node, {}).get('prev', [])
                valid_prev = [p for p in prev_nodes if p not in self.token_block_ids]
                if valid_prev:
                    entry_start = valid_prev[0]
        else: # 'IGP'
            candidates = []
            for tid in self.token_block_ids:
                next_nodes = track_map.get(tid, {}).get('next', [])
                if any(n not in self.token_block_ids for n in next_nodes):
                    candidates.append(tid)
            if candidates:
                gate_node = max(candidates)
                next_nodes = track_map.get(gate_node, {}).get('next', [])
                valid_next = [n for n in next_nodes if n not in self.token_block_ids]
                if valid_next:
                    entry_start = valid_next[0]
                    
        if entry_start is None:
            return []
            
        expected_direction = 'DOWN' if side == 'KSR' else 'UP'
        link_key = 'prev' if side == 'KSR' else 'next'
        max_hops = 8 if side == 'KSR' else 9
        
        current = entry_start
        hops = 0
        result = []
        
        while current is not None and hops < max_hops:
            train = occupied_by_node.get(current)
            if train is None or train.get('direction') != expected_direction:
                break
            result.append(train.get('train_id'))
            
            neighbors = track_map.get(current, {}).get(link_key, [])
            if not neighbors:
                break
            current = neighbors[0]
            hops += 1
            
        return result

    def is_occupied(self) -> bool:
        return len(self.trains_in_block) > 0

    def status(self) -> dict:
        return {
            'direction': self.token_direction,
            'trains_in_block': list(self.trains_in_block),
            'is_free': not self.trains_in_block,
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK VALIDATION — run directly to inspect generated map
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    track_map, loop_sections, end_node, station_nodes, token_blocks = generate_realistic_section()

    print(f"\n{'='*60}")
    print(f"CSMT–Manmad Corridor Map — Validation Report")
    print(f"{'='*60}")
    print(f"Total nodes      : {len(track_map)}")
    print(f"Loop/platform    : {len(loop_sections)}")
    print(f"Token block nodes: {len(token_blocks)} → IDs {token_blocks}")
    print(f"End node         : {end_node}")

    print(f"\n{'─'*60}")
    print(f"Station Anchors:")
    for name, data in station_nodes.items():
        km = data['km']
        n_pf = len(data['platforms'])
        n_lp = len(data['loops'])
        sw_in = data['switch_in']
        sw_out = data['switch_out']
        banker = '🔧 BANKER' if data['is_banker_point'] else ''
        ghat = '⛰️  GHAT START' if data.get('ghat_start') else ('🏁 GHAT END' if data.get('ghat_end') else '')
        print(f"  {name:<10} {km:>4}km  switch_in={sw_in:<4} switch_out={sw_out:<4}  "
              f"platforms={n_pf}  loops={n_lp}  {banker} {ghat}")

    print(f"\n{'─'*60}")
    print(f"Ghat Token Blocks (Kasara→Igatpuri):")
    for tid in token_blocks:
        node = track_map[tid]
        print(f"  node {tid:>4} | km={node['km']:>6.1f} | cap={node['capacity']} | "
              f"speed={node['speed']}kmph | next={node['next']}")

    print(f"\n{'─'*60}")
    print(f"Capacity Distribution:")
    cap_counts = Counter(v['capacity'] for v in track_map.values() if v['type'] not in ['ORIGIN', 'DESTINATION'])
    for cap, count in sorted(cap_counts.items()):
        label = {1: 'Single/Ghat', 2: 'Double line', 4: 'Quadruple line'}.get(cap, f'cap={cap}')
        print(f"  {label:<20}: {count} nodes")

    # Connectivity check — every non-destination node must have at least one next
    broken = [nid for nid, nd in track_map.items()
              if nd['type'] not in ['DESTINATION'] and not nd.get('next')]
    if broken:
        print(f"\n⚠️  BROKEN CONNECTIVITY: nodes with empty next → {broken}")
    else:
        print(f"\n✅ Connectivity OK — all nodes have outgoing edges")

    # Token system smoke test
    token_sys = GhatTokenSystem(token_blocks)
    assert token_sys.can_enter('T1', 'UP') == True
    token_sys.train_entered('T1', 'UP')
    assert token_sys.can_enter('T2', 'UP') == True    # convoy allowed
    assert token_sys.can_enter('T3', 'DOWN') == False  # opposing blocked
    token_sys.train_exited('T1')
    assert token_sys.token_direction is None            # block free
    print(f"✅ GhatTokenSystem smoke test passed")
    print(f"{'='*60}\n")