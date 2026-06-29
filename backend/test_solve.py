import urllib.request, json
req = urllib.request.Request('http://localhost:8000/api/v1/fleet/generate-schedule', data=b'{}', headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as response:
    print(response.read().decode())
