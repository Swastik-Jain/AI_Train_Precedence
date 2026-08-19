import requests
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import compute_deadline

API_URL = "http://localhost:8000/api/v1/fleet"

TRAIN_TYPES = [
    ('Vande Bharat', 'VB', 160),
    ('Rajdhani', 'RJ', 130),
    ('Superfast', 'SF', 110),
    ('Express', 'EX', 100),
    ('Local', 'LC', 80),
    ('Suburban', 'SB', 80),
    ('Passenger', 'PS', 60),
    ('Freight (WAG-9)', 'FR', 75)
]

def populate():
    # First, fetch current trains and delete them to start fresh
    res = requests.get(API_URL)
    if res.ok:
        data = res.json()
        for t in data.get('fleet', []):
            requests.delete(f"{API_URL}/{t['train_id']}")
    
    print("Populating 25 trains...")
    for i in range(1, 26):
        ttype, prefix, speed = random.choice(TRAIN_TYPES)
        start_t = random.randint(0, 50)
        payload = {
            "train_id": f"{prefix}-{100 + i}",
            "train_type": ttype,
            "max_speed": speed,
            "start_time": start_t,
            "deadline": compute_deadline(start_t, speed),
            "direction": random.choice([1, 2])
        }
        r = requests.post(API_URL, json=payload)
        if r.ok:
            print(f"Added {payload['train_id']}")
        else:
            print(f"Failed to add {payload['train_id']}: {r.text}")

if __name__ == "__main__":
    populate()
