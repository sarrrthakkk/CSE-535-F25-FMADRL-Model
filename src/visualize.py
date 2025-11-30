"""
Visualization script for FMADRL training results.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(metrics_path: str) -> list:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_training_results(metrics: list, output_dir: str = None, show: bool = True):
    """Plot training metrics."""
    
    episodes = [m['episode'] for m in metrics]
    rewards = [m['reward'] for m in metrics]
    energy = [m['energy'] for m in metrics]
    battery = [m['final_battery'] * 100 for m in metrics]  # Convert to percentage
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('FMADRL Training Results', fontsize=14, fontweight='bold')
    
    # Plot 1: Rewards over episodes
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, 'b-', alpha=0.3, label='Raw')
    # Moving average
    window = min(10, len(rewards) // 5) if len(rewards) > 5 else 1
    if window > 1:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], rewards_smooth, 'b-', linewidth=2, label=f'Moving Avg ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy consumption
    ax2 = axes[0, 1]
    ax2.plot(episodes, energy, 'r-', alpha=0.3, label='Raw')
    if window > 1:
        energy_smooth = np.convolve(energy, np.ones(window)/window, mode='valid')
        ax2.plot(episodes[window-1:], energy_smooth, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Energy Consumed (mWh)')
    ax2.set_title('Energy Consumption')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final battery level
    ax3 = axes[1, 0]
    ax3.plot(episodes, battery, 'g-', alpha=0.3, label='Raw')
    if window > 1:
        battery_smooth = np.convolve(battery, np.ones(window)/window, mode='valid')
        ax3.plot(episodes[window-1:], battery_smooth, 'g-', linewidth=2, label=f'Moving Avg ({window})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Final Battery (%)')
    ax3.set_title('Battery Remaining After Episode')
    ax3.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Critical (20%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training losses (if available)
    ax4 = axes[1, 1]
    if 'critic_loss' in metrics[0]:
        critic_loss = [m.get('critic_loss', 0) for m in metrics]
        actor_loss = [abs(m.get('actor_loss', 0)) for m in metrics]
        ax4.plot(episodes, critic_loss, 'purple', alpha=0.7, label='Critic Loss')
        ax4.plot(episodes, actor_loss, 'orange', alpha=0.7, label='|Actor Loss|')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Losses')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, 'No loss data available', 
                 ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        output_path = Path(output_dir) / 'training_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def print_summary(metrics: list):
    """Print summary statistics."""
    rewards = [m['reward'] for m in metrics]
    energy = [m['energy'] for m in metrics]
    battery = [m['final_battery'] * 100 for m in metrics]
    
    print("\n" + "="*60)
    print("FMADRL TRAINING SUMMARY")
    print("="*60)
    print(f"\nTotal Episodes: {len(metrics)}")
    print(f"\n{'Metric':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-"*60)
    print(f"{'Reward':<25} {np.mean(rewards):>12.2f} {np.std(rewards):>12.2f} {np.min(rewards):>12.2f} {np.max(rewards):>12.2f}")
    print(f"{'Energy (mWh)':<25} {np.mean(energy):>12.2f} {np.std(energy):>12.2f} {np.min(energy):>12.2f} {np.max(energy):>12.2f}")
    print(f"{'Final Battery (%)':<25} {np.mean(battery):>12.1f} {np.std(battery):>12.1f} {np.min(battery):>12.1f} {np.max(battery):>12.1f}")
    
    # Learning progress (compare first 10% vs last 10%)
    n = len(metrics)
    split = max(1, n // 10)
    
    early_reward = np.mean(rewards[:split])
    late_reward = np.mean(rewards[-split:])
    improvement = ((late_reward - early_reward) / abs(early_reward)) * 100 if early_reward != 0 else 0
    
    print(f"\n{'Learning Progress:':<25}")
    print(f"  Early episodes avg reward: {early_reward:.2f}")
    print(f"  Late episodes avg reward:  {late_reward:.2f}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    early_energy = np.mean(energy[:split])
    late_energy = np.mean(energy[-split:])
    energy_reduction = ((early_energy - late_energy) / early_energy) * 100 if early_energy != 0 else 0
    
    print(f"\n{'Energy Efficiency:':<25}")
    print(f"  Early episodes avg energy: {early_energy:.2f} mWh")
    print(f"  Late episodes avg energy:  {late_energy:.2f} mWh")
    print(f"  Reduction: {energy_reduction:+.1f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize FMADRL training results')
    parser.add_argument('--metrics', type=str, required=True,
                       help='Path to metrics.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (just save)')
    
    args = parser.parse_args()
    
    # Load metrics
    metrics = load_metrics(args.metrics)
    
    # Print summary
    print_summary(metrics)
    
    # Plot results
    plot_training_results(
        metrics, 
        output_dir=args.output or str(Path(args.metrics).parent),
        show=not args.no_show
    )


if __name__ == "__main__":
    main()

