"""
read_tb_metrics.py - Extract key training metrics from TensorBoard event files.
Run from backend/: python read_tb_metrics.py
"""
import sys, os, numpy as np
sys.path.insert(0, '.')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_DIR = 'ai/logs/Phase3'

results = {}
for name in sorted(os.listdir(LOGS_DIR)):
    d = os.path.join(LOGS_DIR, name)
    if not os.path.isdir(d):
        continue
    files = [f for f in os.listdir(d) if 'tfevents' in f]
    if not files:
        continue
    ef = os.path.join(d, files[0])
    ea = EventAccumulator(ef, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    row = {'name': name}
    for tag in ['rollout/ep_rew_mean', 'rollout/ep_len_mean',
                'train/value_loss', 'train/entropy_loss', 'train/approx_kl']:
        if tag in tags:
            vals = [e.value for e in ea.Scalars(tag)]
            row[tag] = {
                'start': vals[0], 'end': vals[-1],
                'max': max(vals), 'min': min(vals), 'n': len(vals)
            }
    results[name] = row

# --- Reward table ---
print('=' * 82)
print('TRAINING REWARD CURVES (rollout/ep_rew_mean)')
print('=' * 82)
h = '{:<35} | {:>10} | {:>10} | {:>10} | {:>4}'.format(
    'RUN NAME', 'START_REW', 'MAX_REW', 'END_REW', 'N')
print(h)
print('-' * 82)
for name, row in results.items():
    if 'rollout/ep_rew_mean' not in row:
        continue
    m = row['rollout/ep_rew_mean']
    trend = 'UP' if m['end'] > m['start'] + 100 else ('DOWN' if m['end'] < m['start'] - 100 else 'FLAT')
    print('{:<35} | {:>10.1f} | {:>10.1f} | {:>10.1f} | {:>4} | {}'.format(
        name, m['start'], m['max'], m['end'], m['n'], trend))
print('=' * 82)

# --- Loss table ---
print()
print('=' * 82)
print('TRAINING DIAGNOSTICS (final values)')
print('=' * 82)
h2 = '{:<35} | {:>13} | {:>12} | {:>10}'.format(
    'RUN NAME', 'VALUE_LOSS', 'ENTROPY', 'APPROX_KL')
print(h2)
print('-' * 82)
for name, row in results.items():
    vl = row['train/value_loss']['end'] if 'train/value_loss' in row else float('nan')
    en = row['train/entropy_loss']['end'] if 'train/entropy_loss' in row else float('nan')
    kl = row['train/approx_kl']['end'] if 'train/approx_kl' in row else float('nan')
    print('{:<35} | {:>13.4f} | {:>12.4f} | {:>10.4f}'.format(name, vl, en, kl))
print('=' * 82)

# --- Episode length table ---
print()
print('=' * 82)
print('EPISODE LENGTHS (rollout/ep_len_mean)')
print('=' * 82)
h3 = '{:<35} | {:>10} | {:>10} | {:>10}'.format(
    'RUN NAME', 'START_LEN', 'MIN_LEN', 'END_LEN')
print(h3)
print('-' * 82)
for name, row in results.items():
    if 'rollout/ep_len_mean' not in row:
        continue
    m = row['rollout/ep_len_mean']
    print('{:<35} | {:>10.1f} | {:>10.1f} | {:>10.1f}'.format(
        name, m['start'], m['min'], m['end']))
print('=' * 82)
