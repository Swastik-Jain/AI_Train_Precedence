import pytest
from main import analyze_simulation, WhatIfScenarioRequest

@pytest.mark.asyncio
async def test_analyze_simulation():
    req = WhatIfScenarioRequest(blocks=[], adjustments=[], forced_actions={"PAS_105": 0})
    res = await analyze_simulation(req)
    assert res is not None, "Analyze simulation should return a result"
    assert "impact" in res, "Result should contain impact data"
    assert "baseline_delay" in res["impact"], "Impact data should contain baseline_delay"
