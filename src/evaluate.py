"""
Evaluate and visualize a trained FMADRL model checkpoint.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from environment import MobileDeviceEnvironment, DEFAULT_SENSORS
from agents import MultiAgentController


def load_checkpoint(checkpoint_path: str, env: MobileDeviceEnvironment) -> MultiAgentController:
    """Load a trained model from checkpoint."""
    controller = MultiAgentController(
        observation_dim=env.observation_dim,
        n_sensors=env.n_sensors,
        hidden_dim=128,
        device="cpu"
    )
    controller.load(checkpoint_path)
    return controller


def run_episode(controller: MultiAgentController, env: MobileDeviceEnvironment, 
                max_steps: int = 1000) -> dict:
    """Run one episode and collect data."""
    obs = env.reset()
    flat_obs = env.get_flat_observation()
    
    # Data collection
    data = {
        "steps": [],
        "rewards": [],
        "battery": [],
        "energy": [],
        "active_sensors": [],
        "sensor_states": [],
        "activity": [],
        "cumulative_reward": [],
        "cumulative_energy": [],
    }
    
    cumulative_reward = 0
    cumulative_energy = 0
    
    for step in range(max_steps):
        # Get action from trained model
        sensor_actions = controller.get_sensor_actions(flat_obs, deterministic=True)
        
        # Step environment
        next_obs, reward, done, info = env.step(sensor_actions)
        next_flat_obs = env.get_flat_observation()
        
        cumulative_reward += reward
        cumulative_energy += info["energy_consumed"]
        
        # Record data
        data["steps"].append(step)
        data["rewards"].append(reward)
        data["battery"].append(info["battery_level"] * 100)
        data["energy"].append(info["energy_consumed"])
        data["active_sensors"].append(info["active_sensors"])
        data["sensor_states"].append(env.sensor_states.copy())
        data["activity"].append(info["current_activity"])
        data["cumulative_reward"].append(cumulative_reward)
        data["cumulative_energy"].append(cumulative_energy)
        
        flat_obs = next_flat_obs
        
        if done:
            break
    
    return data


def visualize_episode(data: dict, sensor_names: list, output_path: str = None):
    """Create visualization of episode."""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('FMADRL Agent Behavior Analysis', fontsize=14, fontweight='bold')
    
    steps = data["steps"]
    
    # 1. Battery level over time
    ax1 = axes[0, 0]
    ax1.plot(steps, data["battery"], 'g-', linewidth=2)
    ax1.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='Critical (20%)')
    ax1.fill_between(steps, data["battery"], alpha=0.3, color='green')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Battery Level (%)')
    ax1.set_title('Battery Depletion Over Time')
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative reward and energy
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    line1, = ax2.plot(steps, data["cumulative_reward"], 'b-', linewidth=2, label='Cumulative Reward')
    line2, = ax2_twin.plot(steps, data["cumulative_energy"], 'r-', linewidth=2, label='Cumulative Energy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Reward', color='blue')
    ax2_twin.set_ylabel('Cumulative Energy (mWh)', color='red')
    ax2.set_title('Reward vs Energy Trade-off')
    ax2.legend(handles=[line1, line2], loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Active sensors over time
    ax3 = axes[1, 0]
    ax3.plot(steps, data["active_sensors"], 'purple', linewidth=1.5)
    ax3.fill_between(steps, data["active_sensors"], alpha=0.3, color='purple')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Number of Active Sensors')
    ax3.set_title('Sensor Activation Over Time')
    ax3.set_ylim(0, len(sensor_names) + 0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Sensor activation heatmap
    ax4 = axes[1, 1]
    sensor_matrix = np.array(data["sensor_states"]).T
    im = ax4.imshow(sensor_matrix, aspect='auto', cmap='RdYlGn', 
                    interpolation='nearest', vmin=0, vmax=1)
    ax4.set_yticks(range(len(sensor_names)))
    ax4.set_yticklabels(sensor_names)
    ax4.set_xlabel('Step')
    ax4.set_title('Sensor Activation Pattern (Green=On, Red=Off)')
    plt.colorbar(im, ax=ax4, label='Active')
    
    # 5. Instant reward over time
    ax5 = axes[2, 0]
    ax5.plot(steps, data["rewards"], 'b-', alpha=0.5, linewidth=0.5)
    # Moving average
    window = min(50, len(steps) // 5) if len(steps) > 5 else 1
    if window > 1:
        rewards_smooth = np.convolve(data["rewards"], np.ones(window)/window, mode='valid')
        ax5.plot(steps[window-1:], rewards_smooth, 'b-', linewidth=2, label=f'Moving Avg ({window})')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Instant Reward')
    ax5.set_title('Reward Per Step')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. User activity distribution
    ax6 = axes[2, 1]
    activities = data["activity"]
    unique_activities = list(set(activities))
    activity_counts = [activities.count(a) for a in unique_activities]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_activities)))
    bars = ax6.bar(unique_activities, activity_counts, color=colors)
    ax6.set_xlabel('Activity')
    ax6.set_ylabel('Time Steps')
    ax6.set_title('User Activity Distribution')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    return fig


def print_episode_summary(data: dict, sensor_names: list):
    """Print summary statistics for the episode."""
    print("\n" + "="*60)
    print("EPISODE EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nDuration: {len(data['steps'])} steps")
    print(f"Final Battery: {data['battery'][-1]:.1f}%")
    print(f"Total Reward: {data['cumulative_reward'][-1]:.2f}")
    print(f"Total Energy: {data['cumulative_energy'][-1]:.2f} mWh")
    
    # Sensor usage statistics
    sensor_states = np.array(data["sensor_states"])
    print(f"\n{'Sensor':<20} {'Active %':>10} {'Avg Power (mW)':>15}")
    print("-"*45)
    
    for i, name in enumerate(sensor_names):
        active_pct = sensor_states[:, i].mean() * 100
        print(f"{name:<20} {active_pct:>10.1f}%")
    
    # Activity breakdown
    print(f"\nUser Activity Breakdown:")
    activities = data["activity"]
    for activity in set(activities):
        count = activities.count(activity)
        pct = count / len(activities) * 100
        print(f"  {activity}: {count} steps ({pct:.1f}%)")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate FMADRL checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pt)')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of steps to run')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')
    
    args = parser.parse_args()
    
    # Create environment
    env = MobileDeviceEnvironment(episode_length=args.steps)
    sensor_names = [s.name for s in DEFAULT_SENSORS]
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    controller = load_checkpoint(args.checkpoint, env)
    
    # Run episode
    print(f"Running evaluation for {args.steps} steps...")
    data = run_episode(controller, env, max_steps=args.steps)
    
    # Print summary
    print_episode_summary(data, sensor_names)
    
    # Visualize
    output_path = args.output or str(Path(args.checkpoint).parent / "checkpoint_evaluation.png")
    
    if not args.no_show:
        visualize_episode(data, sensor_names, output_path)
    else:
        visualize_episode(data, sensor_names, output_path)
        plt.close()


if __name__ == "__main__":
    main()

