import asyncio
import json
from httpx import Client

c = Client()
try:
    r = c.get("http://localhost:8000/api/v1/topology")
    data = r.json()
    for t in data.get("trains", []):
        print(f"ID: {t['train_id']} | Status: {t['status']} | pos: {t.get('position_percentage')} | speed: {t.get('speed_kmh')}")
except Exception as e:
    print(e)
