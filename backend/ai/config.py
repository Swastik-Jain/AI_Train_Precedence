# ai/config.py — CSMT → Manmad Corridor
# Real train fleet based on Central Railway Bhusawal-Kalyan division
# Timetable data sourced from NTES / indiarailinfo

import random

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAX_TRAINS_CAPACITY = 25        # Phase 4: 25 trains — real congestion ceiling for L6 curriculum
MAX_SPEED           = 130       # km/h — fastest train on corridor (Rajdhani)
SECTION_LENGTH_KM   = 261       # CSMT → Manmad total distance

# Evaluation Stress Test
DEADLINE_MULTIPLIER = 0.45      # 45% of normal = tight but achievable

# Physics
ACCEL_RATE          = 10        # km/h gained per sim-step
DECEL_RATE          = 20        # km/h lost per sim-step
GHAT_SPEED_UP       = 50        # max speed inside ghat (uphill)
GHAT_SPEED_DOWN     = 60        # max speed inside ghat (downhill)

# Dwell times (sim-steps)
DWELL_TIME_PLATFORM = 3         # scheduled stop at station platform
DWELL_TIME_LOOP     = 0         # holding in loop — no dwell, just waiting
BANKER_ATTACH_TIME  = 4         # steps for banker loco to couple at Kasara
BANKER_DETACH_TIME  = 2         # steps for banker loco to uncouple at Igatpuri

# Observation normalization — km-based, not block-count-based
DANGER_HORIZON_KM   = 10.0      # braking horizon for safety features
SPACING_HORIZON_KM  = 30.0      # operational horizon for headway features
SECTION_HORIZON_KM  = SECTION_LENGTH_KM  # full section for opposing train distance

# ─────────────────────────────────────────────────────────────────────────────
# STATION ORDER — canonical corridor sequence (DOWN direction: CSMT→Manmad)
# ─────────────────────────────────────────────────────────────────────────────
STATION_ORDER_DOWN = ['CSMT', 'DADAR', 'KALYAN', 'KASARA', 'IGATPURI', 'DEVLALI', 'NASHIK', 'MANMAD']
STATION_ORDER_UP   = list(reversed(STATION_ORDER_DOWN))

# ─────────────────────────────────────────────────────────────────────────────
# REAL TRAIN ARCHETYPES
# Based on actual trains running on CSMT-Manmad corridor
# ─────────────────────────────────────────────────────────────────────────────
#
# direction : 'UP'   = Manmad → CSMT
#             'DOWN' = CSMT → Manmad
#
# stops     : stations where this train makes a scheduled halt
#             trains pass through other stations without stopping
#             (pass-through = no dwell, reduced speed only)
#
# banker_required : True  = must attach banker at Kasara (UP) or Igatpuri (DOWN)
#                  False = push-pull loco, skips banker stop entirely (Rajdhani only)
#
# priority  : 6=highest (Rajdhani), 1=lowest (heavy freight)
#             higher priority trains get route preference in conflicts

TRAIN_ARCHETYPES = [

    # ── Priority 6: Rajdhani Express ──────────────────────────────────────
    # CR Rajdhani (22221/22222) — push-pull WAP-7, no banker needed
    # DOWN: departs CSMT 16:35, arrives Manmad ~21:10 (real schedule)
    # Stops: CSMT, KALYAN, IGATPURI (technical, no banker), NASHIK, MANMAD
    {
        'archetype': 'RAJDHANI',
        'priority': 10,
        'max_speed': 130,
        'accel_rate': 15,
        'decel_rate': 25,
        'banker_required': False,   # push-pull WAP-7
        'stops_down': ['CSMT', 'KALYAN', 'IGATPURI', 'NASHIK', 'MANMAD'],
        'stops_up':   ['MANMAD', 'NASHIK', 'IGATPURI', 'KALYAN', 'CSMT'],
        'weight': 1,                # 1 per direction in schedule generator
    },

    # ── Priority 5: Superfast Express ─────────────────────────────────────
    # Panchvati Express (12109/12110) — daily CSMT-Manmad
    # DOWN: departs CSMT 18:15, arrives Manmad 22:50
    # UP:   departs Manmad 06:02, arrives CSMT 10:45
    # Stops: CSMT, DADAR, KALYAN, IGATPURI, DEVLALI, NASHIK, MANMAD
    {
        'archetype': 'SUPERFAST',
        'priority': 8,
        'max_speed': 110,
        'accel_rate': 12,
        'decel_rate': 22,
        'banker_required': True,
        'stops_down': ['CSMT', 'DADAR', 'KALYAN', 'KASARA', 'IGATPURI', 'DEVLALI', 'NASHIK', 'MANMAD'],
        'stops_up':   ['MANMAD', 'NASHIK', 'DEVLALI', 'IGATPURI', 'KASARA', 'KALYAN', 'DADAR', 'CSMT'],
        'weight': 2,
    },

    # ── Priority 4: Mail/Express ───────────────────────────────────────────
    # Pushpak Express (12533/12534) — Lucknow-CSMT via Manmad
    # Passes through our corridor: Manmad→Nashik→Igatpuri→Kalyan→CSMT
    # Stops: MANMAD, NASHIK, IGATPURI, KALYAN, DADAR, CSMT
    {
        'archetype': 'MAIL_EXPRESS',
        'priority': 6,
        'max_speed': 110,
        'accel_rate': 10,
        'decel_rate': 20,
        'banker_required': True,
        'stops_down': ['CSMT', 'DADAR', 'KALYAN', 'KASARA', 'IGATPURI', 'NASHIK', 'MANMAD'],
        'stops_up':   ['MANMAD', 'NASHIK', 'IGATPURI', 'KASARA', 'KALYAN', 'DADAR', 'CSMT'],
        'weight': 2,
    },

    # ── Priority 3: Passenger Express ─────────────────────────────────────
    # Stops at every station — the slow mover that causes overtaking conflicts
    {
        'archetype': 'PASSENGER',
        'priority': 3,
        'max_speed': 75,
        'accel_rate': 8,
        'decel_rate': 15,
        'banker_required': True,
        'stops_down': ['CSMT', 'DADAR', 'KALYAN', 'KASARA', 'IGATPURI', 'DEVLALI', 'NASHIK', 'MANMAD'],
        'stops_up':   ['MANMAD', 'NASHIK', 'DEVLALI', 'IGATPURI', 'KASARA', 'KALYAN', 'DADAR', 'CSMT'],
        'weight': 2,
    },

    # ── Priority 1: Goods/Freight ──────────────────────────────────────────
    # Slow, heavy, uses loops to let express trains overtake
    # Stops at major stations only — uses loop sidings at small stations
    {
        'archetype': 'GOODS',
        'priority': 2,
        'max_speed': 60,
        'accel_rate': 5,
        'decel_rate': 10,
        'banker_required': True,
        'stops_down': ['CSMT', 'KALYAN', 'KASARA', 'IGATPURI', 'NASHIK', 'MANMAD'],
        'stops_up':   ['MANMAD', 'NASHIK', 'IGATPURI', 'KASARA', 'KALYAN', 'CSMT'],
        'weight': 2,
    },

    # ── Priority 0: Banker Locomotive ─────────────────────────────────────
    # Special type — only runs Kasara ↔ Igatpuri
    # UP run: attaches at Kasara, pushes train to Igatpuri, detaches
    # DOWN run: runs light engine from Igatpuri back to Kasara
    # Always occupies token block — agent must schedule around it
    {
        'archetype': 'BANKER',
        'priority': 0,
        'max_speed': 50,
        'accel_rate': 8,
        'decel_rate': 15,
        'banker_required': False,   # IS the banker
        'stops_down': ['IGATPURI', 'KASARA'],   # light engine return
        'stops_up':   ['KASARA', 'IGATPURI'],   # pushing UP train
        'weight': 0,                # not randomly spawned — spawned by ghat logic
    },
]

# Quick lookup by archetype name
ARCHETYPE_BY_NAME = {a['archetype']: a for a in TRAIN_ARCHETYPES}

# Weighted pool for random schedule generation (excludes BANKER — spawned separately)
_SPAWN_POOL = [
    a for a in TRAIN_ARCHETYPES
    for _ in range(a['weight'])
    if a['archetype'] != 'BANKER'
]

# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_schedule(num_trains: int = 5, seed: int = None):
    """
    Generate a randomized but realistic fleet + schedule for one episode.

    Parameters
    ----------
    num_trains : int
        Total trains to spawn (banker locos added automatically on top).
        Curriculum: 2 → 5 → 7 → 10

    seed : int | None
        Fix for reproducible eval episodes (e.g. baseline comparison).

    Returns
    -------
    fleet    : list[dict]   — train definitions
    schedule : dict         — {train_id: {start_time, deadline, stops, direction}}
    """
    rng = random.Random(seed)

    fleet    = []
    schedule = {}

    # ── Phase 1: Micro-symmetry for initial curriculum (2 trains) ─────────
    if num_trains == 2:
        fleet = [
            {
                'id':             'GOODS_100',
                'archetype':      'GOODS',
                'priority':       1,
                'direction':      'DOWN',
                'max_speed':      60,
                'accel_rate':     5,
                'decel_rate':     10,
                'banker_required': True,
                'position':       0,
                'speed':          0,
                'target_speed':   0,
                'delay':          0,
                'idle_time':      0,
                'dwell_rem':      0,
                'finished':       False,
                'visited_nodes':  set(),
                'banker_attached': False,
                'banker_wait':    0,
            },
            {
                'id':             'SF_101',
                'archetype':      'SUPERFAST',
                'priority':       5,
                'direction':      'UP',
                'max_speed':      110,
                'accel_rate':     12,
                'decel_rate':     22,
                'banker_required': True,
                'position':       0,
                'speed':          0,
                'target_speed':   0,
                'delay':          0,
                'idle_time':      0,
                'dwell_rem':      0,
                'finished':       False,
                'visited_nodes':  set(),
                'banker_attached': False,
                'banker_wait':    0,
            },
        ]
        # Deadlines must be achievable at the train's actual max_speed.
        # GOODS at 60km/h over 261km ≈ 261 steps (1km/step at 60km/h).
        # SUPERFAST at 110km/h over 261km ≈ 143 steps.
        # Add generous buffer (×2.5) so early training doesn't drown in
        # deadline penalties before the model learns to move at all.
        schedule = {
            'GOODS_100': {
                'start_time': 0,
                'deadline':   600,   # ~2.5× realistic travel time for GOODS
                'stops':      ARCHETYPE_BY_NAME['GOODS']['stops_down'],
                'direction':  'DOWN',
            },
            'SF_101': {
                'start_time': 15,
                'deadline':   400,   # ~2.5× realistic travel time for SUPERFAST
                'stops':      ARCHETYPE_BY_NAME['SUPERFAST']['stops_up'],
                'direction':  'UP',
            },
        }
        return fleet, schedule

    # ── Phase 2: Mixed fleet for curriculum stages 5, 7, 10+ ─────────────
    # Hard floor: at least floor(n/2) trains per direction before shuffle.
    # The old i%2 alternation was correct in principle but rng.shuffle()
    # could still produce skewed batches (e.g. 4 DOWN + 1 UP for 5 trains).
    # This guarantee ensures every episode has genuine bidirectional conflict.
    n_up   = num_trains // 2
    n_down = num_trains - n_up
    directions = ['UP'] * n_up + ['DOWN'] * n_down
    rng.shuffle(directions)

    current_time = 0

    for i, direction in enumerate(directions):
        archetype_data = rng.choice(_SPAWN_POOL)
        arch_name = archetype_data['archetype']

        t_id = f"{arch_name[:3].upper()}_{100 + i}"

        # Staggered departure — realistic headways
        interval = rng.randint(8, 15)   # minutes between trains
        current_time += interval

        # Deadline based on realistic travel time for this train type
        # Fast trains: ~4.5 hrs real time = ~270 sim-steps at our resolution
        # Slow trains: ~6 hrs real time = ~360 sim-steps
        if archetype_data['priority'] >= 5:
            travel_budget = rng.randint(220, 280)
        elif archetype_data['priority'] >= 3:
            travel_budget = rng.randint(280, 340)
        else:
            travel_budget = rng.randint(340, 420)   # freight gets more time

        deadline = current_time + int(travel_budget * DEADLINE_MULTIPLIER)

        stops = (archetype_data['stops_down']
                 if direction == 'DOWN'
                 else archetype_data['stops_up'])

        train = {
            'id':              t_id,
            'archetype':       arch_name,
            'priority':        archetype_data['priority'],
            'direction':       direction,
            'max_speed':       archetype_data['max_speed'],
            'accel_rate':      archetype_data['accel_rate'],
            'decel_rate':      archetype_data['decel_rate'],
            'banker_required': archetype_data['banker_required'],

            # Runtime state — reset each episode
            'position':        0,
            'speed':           0,
            'target_speed':    0,
            'delay':           0,
            'idle_time':       0,
            'dwell_rem':       0,
            'finished':        False,
            'visited_nodes':   set(),

            # Banker state
            'banker_attached': False,
            'banker_wait':     0,       # steps remaining for attach/detach
        }

        fleet.append(train)
        schedule[t_id] = {
            'start_time': current_time,
            'deadline':   deadline,
            'stops':      stops,
            'direction':  direction,
        }

    return fleet, schedule


def generate_stress_schedule(num_trains: int = 10, seed: int = None):
    """
    Stress-test schedule: clustered spawns + tight deadlines.
    Forces 4-5 trains into the ghat section simultaneously from
    both directions, creating genuine queueing conflicts where
    dispatcher ordering truly matters.

    NOTE: num_trains < 4 falls back to normal schedule because stress
    mode with 2-3 trains can produce same-direction pairs (no ghat
    conflict) that run to full 1500-step timeout with -4000 rewards,
    completely swamping the training signal.

    Key differences from normal schedule:
      - Trains spawn in tight bursts (2-4 step intervals)
      - Deadlines are 50% of normal (very tight)
      - Direction clustering: bursts alternate UP/DOWN
    """
    # Guard: stress mode needs ≥4 trains to guarantee bidirectional bursts.
    # With 2-3 trains the burst logic often produces same-direction pairs
    # which run to a full 1500-step timeout with -4000 rewards, completely
    # swamping the learning signal at early curriculum levels.
    if num_trains < 4:
        return generate_daily_schedule(num_trains, seed=seed)

    rng = random.Random(seed)

    fleet    = []
    schedule = {}

    # Cluster trains into bursts of 3-5 with opposing directions
    remaining = num_trains
    current_time = 0
    burst_id = 0

    while remaining > 0:
        burst_size = min(remaining, rng.randint(3, 5))
        # Alternate burst directions — first burst DOWN, next UP, etc.
        burst_dir = 'DOWN' if burst_id % 2 == 0 else 'UP'
        # Mix in 1-2 opposing trains per burst for maximum conflict
        directions = [burst_dir] * burst_size
        if burst_size >= 3:
            n_opposing = rng.randint(1, 2)
            for j in range(n_opposing):
                directions[-(j+1)] = 'UP' if burst_dir == 'DOWN' else 'DOWN'
        rng.shuffle(directions)

        for j, direction in enumerate(directions):
            archetype_data = rng.choice(_SPAWN_POOL)
            arch_name = archetype_data['archetype']
            t_id = f"{arch_name[:3].upper()}_{100 + len(fleet)}"

            # Tight cluster: 2-4 steps between trains in same burst
            interval = rng.randint(2, 4)
            current_time += interval

            # Very tight deadlines (50% of normal)
            if archetype_data['priority'] >= 5:
                travel_budget = rng.randint(220, 280)
            elif archetype_data['priority'] >= 3:
                travel_budget = rng.randint(280, 340)
            else:
                travel_budget = rng.randint(340, 420)

            deadline = current_time + int(travel_budget * DEADLINE_MULTIPLIER)

            stops = (archetype_data['stops_down']
                     if direction == 'DOWN'
                     else archetype_data['stops_up'])

            train = {
                'id':              t_id,
                'archetype':       arch_name,
                'priority':        archetype_data['priority'],
                'direction':       direction,
                'max_speed':       archetype_data['max_speed'],
                'accel_rate':      archetype_data['accel_rate'],
                'decel_rate':      archetype_data['decel_rate'],
                'banker_required': archetype_data['banker_required'],
                'position':        0,
                'speed':           0,
                'target_speed':    0,
                'delay':           0,
                'idle_time':       0,
                'dwell_rem':       0,
                'finished':        False,
                'visited_nodes':   set(),
                'banker_attached': False,
                'banker_wait':     0,
            }

            fleet.append(train)
            schedule[t_id] = {
                'start_time': current_time,
                'deadline':   deadline,
                'stops':      stops,
                'direction':  direction,
            }

        remaining -= burst_size
        burst_id += 1
        # Gap between bursts — small enough that first burst hasn't cleared ghat
        current_time += rng.randint(8, 15)

    return fleet, schedule


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS — used by env before curriculum overrides
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_FLEET, SCHEDULE = generate_daily_schedule(num_trains=2)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Config Validation — CSMT-Manmad Fleet Generator")
    print("="*60)

    for n in [2, 5, 7, 10]:
        fleet, sched = generate_daily_schedule(num_trains=n, seed=42)
        directions = [t['direction'] for t in fleet]
        up   = directions.count('UP')
        down = directions.count('DOWN')
        archetypes = [t['archetype'] for t in fleet]
        print(f"\n  num_trains={n:>2} | UP={up} DOWN={down} | {archetypes}")
        for t in fleet:
            s = sched[t['id']]
            print(f"    {t['id']:<12} dir={t['direction']:<5} "
                  f"prio={t['priority']} speed={t['max_speed']:>3}kmph "
                  f"banker={str(t['banker_required']):<5} "
                  f"start={s['start_time']:>3} deadline={s['deadline']:>4} "
                  f"stops={s['stops']}")

    # Verify banker archetype exists and is not in spawn pool
    assert 'BANKER' not in [a['archetype'] for a in _SPAWN_POOL], \
        "BANKER should never be in random spawn pool"

    # Verify all archetypes have both up and down stops
    for a in TRAIN_ARCHETYPES:
        if a['archetype'] == 'BANKER':
            continue
        assert 'CSMT' in a['stops_down'] and 'MANMAD' in a['stops_down'], \
            f"{a['archetype']} stops_down missing terminus"
        assert 'MANMAD' in a['stops_up'] and 'CSMT' in a['stops_up'], \
            f"{a['archetype']} stops_up missing terminus"

    print("\n" + "="*60)
    print("✅ All validations passed")
    print("="*60)