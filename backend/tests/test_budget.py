from ai.config import generate_stress_schedule

def test_generate_stress_schedule():
    fleet, schedule = generate_stress_schedule(10)
    assert len(fleet) == 10, "Fleet should contain exactly 10 trains"
    assert len(schedule) > 0, "Schedule should not be empty"
