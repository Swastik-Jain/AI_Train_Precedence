import os
import sys
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Constants
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

def get_log_dirs(root_dir):
    """Recursively find all directories containing tfevents files."""
    log_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "tfevents" in file:
                log_dirs.append(root)
                break 
    return log_dirs

def analyze_log(log_dir):
    """Extract metrics from all tfevents files in a directory."""
    files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    
    if not files:
        return None

    combined_metrics = defaultdict(list)
    
    print(f"   📂 Found {len(files)} event files in {os.path.basename(log_dir)}")

    for file in files:
        event_file = os.path.join(log_dir, file)
        try:
            ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
            ea.Reload()
            
            # Check for tags
            tags = ea.Tags()['scalars']
            if 'rollout/ep_rew_mean' in tags:
                events = ea.Scalars('rollout/ep_rew_mean')
                for e in events:
                    combined_metrics['rollout/ep_rew_mean'].append(e.value)
                    
        except Exception as e:
            print(f"   ⚠️ Could not read {file}: {e}")

    results = {}
    if combined_metrics['rollout/ep_rew_mean']:
        values = combined_metrics['rollout/ep_rew_mean']
        results['rollout/ep_rew_mean'] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'start': values[0],
            'end': values[-1],
            'count': len(values)
        }
    
    return results

def print_summary(results):
    """Print a detailed summary table."""
    print("\n" + "="*100)
    print(f"{'MODEL NAME':<40} | {'START':<10} | {'MAX':<10} | {'END':<10} | {'TREND':<10}")
    print("-" * 100)
    
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    for name, metrics in sorted_results:
        if 'rollout/ep_rew_mean' in metrics:
            m = metrics['rollout/ep_rew_mean']
            
            trend = "➡️"
            if m['end'] > m['start'] + 100: trend = "↗️ UP"
            elif m['end'] < m['start'] - 100: trend = "↘️ DOWN"
            
            print(f"{name:<40} | {m['start']:<10.2f} | {m['max']:<10.2f} | {m['end']:<10.2f} | {trend:<10}")
        else:
            print(f"{name:<40} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | -")

    print("="*100 + "\n")

def main():
    print(f"🔍 Scanning for logs in: {LOGS_DIR}")
    
    if not os.path.exists(LOGS_DIR):
        print("❌ Logs directory not found.")
        return

    log_dirs = get_log_dirs(LOGS_DIR)
    results = {}

    for d in log_dirs:
        # Get relative path as name
        rel_name = os.path.relpath(d, LOGS_DIR)
        print(f"   Processing: {rel_name}...")
        
        metrics = analyze_log(d)
        if metrics:
            results[rel_name] = metrics
            
    print_summary(results)

if __name__ == "__main__":
    main()
