import requests
import json

resp = requests.post("http://localhost:8000/api/v1/simulation/analyze", json={"latencies":{}, "forced_actions":{}})
print("Baseline outcomes:", json.dumps(resp.json(), indent=2))
