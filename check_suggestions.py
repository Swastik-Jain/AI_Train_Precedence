import requests
import json
print("Waiting 5s to let inference run...")
import time; time.sleep(5)
res = requests.get('http://localhost:8000/api/v1/telemetry')
if res.status_code == 200:
    print("Telemetry fetched!")
else:
    print("Failed")
