import sys
sys.path.append('.')
from ai.config import generate_stress_schedule
fleet, schedule = generate_stress_schedule(10)
print(schedule)
