import requests
import json
from datetime import datetime, timezone, timedelta

now = datetime.now(timezone.utc)
end = now + timedelta(hours=2)

block = {
    "element_id": "edge-1-2",
    "type": "TRACK_SEGMENT",
    "start_time": now.isoformat(),
    "end_time": end.isoformat(),
    "severity": "TOTAL_BLOCK",
    "reason": "Test"
}
requests.post("http://localhost:8000/api/v1/sandbox/blocks", json=block)

payload = {
    "latencies": {},
    "forced_actions": {}
}

resp = requests.post("http://localhost:8000/api/v1/simulation/analyze", json=payload)
print("With Sandbox Block:")
print(json.dumps(resp.json().get('impact'), indent=2))
