import requests
import json
import time

resp = requests.post("http://localhost:8000/api/v1/simulation/analyze", json={"latencies":{}, "forced_actions":{}})
impact1 = resp.json().get('impact')
print("Before Block:", json.dumps(impact1, indent=2))

# get active trains to find an edge that is busy
resp_fleet = requests.get("http://localhost:8000/api/v1/fleet")
trains = resp_fleet.json().get("trains", [])
if not trains:
    print("No active trains!")
else:
    edge_id = trains[0]["edge_id"]
    print("Blocking edge:", edge_id)
    block = {
        "element_id": edge_id,
        "type": "TRACK_SEGMENT",
        "start_time": "2000-01-01T00:00:00Z",
        "end_time": "2099-01-01T00:00:00Z",
        "severity": "TOTAL_BLOCK",
        "reason": "Test"
    }
    requests.post("http://localhost:8000/api/v1/sandbox/blocks", json=block)

    resp2 = requests.post("http://localhost:8000/api/v1/simulation/analyze", json={"latencies":{}, "forced_actions":{}})
    impact2 = resp2.json().get('impact')
    print("After Block:", json.dumps(impact2, indent=2))

    # Clean up
    requests.delete(f"http://localhost:8000/api/v1/sandbox/blocks/{edge_id}")
