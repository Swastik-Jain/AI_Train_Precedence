import urllib.request
import json
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
    try:
        req = urllib.request.Request(API_URL)
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode())
            for t in data.get('fleet', []):
                req_del = urllib.request.Request(f"{API_URL}/{t['train_id']}", method='DELETE')
                urllib.request.urlopen(req_del)
    except Exception as e:
        print(f"Note: Could not fetch/delete existing trains (they might already be empty): {e}")
    
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
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req_post = urllib.request.Request(API_URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req_post) as res:
                print(f"Added {payload['train_id']}")
        except Exception as e:
            print(f"Failed to add {payload['train_id']}: {e}")

if __name__ == "__main__":
    populate()
