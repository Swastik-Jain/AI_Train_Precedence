import pytest
from backend.config import compute_deadline

def test_compute_deadline_achievable():
    # Real minimum transit steps for a 110 km/h train across 261km is ~143.
    # We want to assert the deadline provides a realistic buffer over the bare minimum.
    deadline = compute_deadline(0, 110)
    assert deadline > 143, f"Deadline {deadline} is too tight or equal to bare minimum transit time."
    assert deadline == 356, f"Expected 356 but got {deadline}"

def test_compute_deadline_zero_speed():
    with pytest.raises(ValueError):
        compute_deadline(0, 0)
