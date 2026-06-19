import requests
import time
res = requests.get('http://localhost:8000/api/v1/system/inference-status')
print(res.json())
