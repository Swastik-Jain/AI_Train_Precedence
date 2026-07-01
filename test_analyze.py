import sys
import asyncio
sys.path.insert(0, ".")
from main import analyze_simulation, WhatIfScenarioRequest

async def test():
    req = WhatIfScenarioRequest(blocks=[], adjustments=[], forced_actions={"PAS_105": 0})
    res = await analyze_simulation(req)
    print("Success")

asyncio.run(test())
