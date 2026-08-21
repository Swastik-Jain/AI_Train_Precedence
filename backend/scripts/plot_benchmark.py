import json
import os
import numpy as np
import matplotlib.pyplot as plt

def generate_charts(json_path='results/benchmark_table.json', out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    labels = [r['baseline'] for r in data]
    
    # 1. Safety Comparison (Collisions & Timeouts)
    collisions = [r['collision_rate'] for r in data]
    timeouts = [r['timeout_rate'] for r in data]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, collisions, width, label='Collision Rate (%)', color='#e74c3c')
    ax.bar(x + width/2, timeouts, width, label='Timeout Rate (%)', color='#f39c12')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Safety & Reliability Comparison (Lower is Better)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for i, (c, t) in enumerate(zip(collisions, timeouts)):
        ax.text(i - width/2, c + 1, f'{c}%', ha='center', va='bottom')
        ax.text(i + width/2, t + 1, f'{t}%', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'safety_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Delay Comparison (Average Delay & Ghat Wait)
    avg_delay = [r['avg_delay'] for r in data]
    ghat_wait = [r['avg_ghat_wait'] for r in data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, avg_delay, width, label='Avg Train Delay (Steps)', color='#3498db')
    ax.bar(x + width/2, ghat_wait, width, label='Ghat Bottleneck Wait (Steps)', color='#9b59b6')
    
    ax.set_ylabel('Simulation Steps', fontsize=12)
    ax.set_title('Traffic Flow & Delay Comparison (Lower is Better)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars
    for i, (d, g) in enumerate(zip(avg_delay, ghat_wait)):
        ax.text(i - width/2, d + 10, f'{d:.1f}', ha='center', va='bottom')
        ax.text(i + width/2, g + 10, f'{g:.1f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'delay_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Reward Comparison
    avg_reward = [r['avg_reward'] for r in data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#95a5a6' if i < 2 else ('#2ecc71' if 'Shield' in labels[i] else '#27ae60') for i in range(len(labels))]
    bars = ax.bar(x, avg_reward, 0.6, color=colors)
    
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Overall Agent Performance Reward (Higher is Better)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - (abs(height)*0.05) if height < 0 else height + 5,
                f'{height:.1f}', ha='center', va='bottom' if height >=0 else 'top', color='black')
                
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'reward_comparison.png'), dpi=300)
    plt.close()
    
    print("Successfully generated charts in", out_dir)

if __name__ == '__main__':
    generate_charts()
