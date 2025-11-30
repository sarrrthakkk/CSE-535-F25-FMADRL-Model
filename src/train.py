"""
FMADRL Training Script

Main training loop for Federated Multi-Agent Deep Reinforcement Learning
for mobile energy optimization.
"""

import argparse
import os
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

from environment import MobileDeviceEnvironment, MultiDeviceEnvironment
from agents import MultiAgentController
from federated import FederatedAggregator, FederatedClient, TransferLearning


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FMADRL for Mobile Energy Optimization"
    )
    
    # Training settings
    parser.add_argument("--n-devices", type=int, default=10,
                       help="Number of simulated devices")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of training episodes")
    parser.add_argument("--episode-length", type=int, default=1000,
                       help="Steps per episode")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    
    # Federated learning settings
    parser.add_argument("--fed-rounds", type=int, default=50,
                       help="Number of federated learning rounds")
    parser.add_argument("--client-fraction", type=float, default=0.5,
                       help="Fraction of clients per round")
    parser.add_argument("--local-epochs", type=int, default=5,
                       help="Local training epochs per round")
    parser.add_argument("--dp", action="store_true",
                       help="Enable differential privacy")
    parser.add_argument("--dp-epsilon", type=float, default=1.0,
                       help="DP epsilon")
    
    # Model settings
    parser.add_argument("--hidden-dim", type=int, default=128,
                       help="Hidden layer dimension")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--save-freq", type=int, default=10,
                       help="Save model every N episodes")
    parser.add_argument("--eval-freq", type=int, default=5,
                       help="Evaluate every N episodes")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to use")
    
    return parser.parse_args()


def create_output_dir(base_dir: str) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def evaluate_policy(
    controller: MultiAgentController,
    env: MobileDeviceEnvironment,
    n_episodes: int = 5
) -> Dict[str, float]:
    """Evaluate current policy."""
    total_rewards = []
    total_energy = []
    final_battery = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        flat_obs = env.get_flat_observation()
        episode_reward = 0
        
        while True:
            # Get deterministic actions
            sensor_actions = controller.get_sensor_actions(flat_obs, deterministic=True)
            
            obs, reward, done, info = env.step(sensor_actions)
            flat_obs = env.get_flat_observation()
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_energy.append(env.total_energy_consumed)
        final_battery.append(env.battery_level)
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_energy": np.mean(total_energy),
        "mean_final_battery": np.mean(final_battery),
    }


def train_single_device(args):
    """Train on a single device (baseline)."""
    print("=" * 60)
    print("FMADRL Single Device Training")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Create environment
    env = MobileDeviceEnvironment(
        episode_length=args.episode_length
    )
    
    # Create controller
    controller = MultiAgentController(
        observation_dim=env.observation_dim,
        n_sensors=env.n_sensors,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device
    )
    
    # Training metrics
    metrics_history = []
    
    # Training loop
    print("\nStarting training...")
    for episode in tqdm(range(args.episodes), desc="Episodes"):
        obs = env.reset()
        flat_obs = env.get_flat_observation()
        episode_reward = 0
        episode_metrics = []
        
        for step in range(args.episode_length):
            # Get actions
            sensor_actions = controller.get_sensor_actions(flat_obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(sensor_actions)
            next_flat_obs = env.get_flat_observation()
            
            # Store transition
            controller.store_transition(
                flat_obs, sensor_actions, reward, next_flat_obs, done
            )
            
            # Update policy
            if len(controller.replay_buffer) >= args.batch_size:
                train_metrics = controller.update(batch_size=args.batch_size)
                if train_metrics:
                    episode_metrics.append(train_metrics)
            
            flat_obs = next_flat_obs
            episode_reward += reward
            
            if done:
                break
        
        # Log episode metrics
        metrics = {
            "episode": episode,
            "reward": episode_reward,
            "steps": step + 1,
            "energy": env.total_energy_consumed,
            "final_battery": env.battery_level,
        }
        
        if episode_metrics:
            metrics["critic_loss"] = np.mean([m.get("critic_loss", 0) for m in episode_metrics])
            metrics["actor_loss"] = np.mean([m.get("actor_loss_mean", 0) for m in episode_metrics])
        
        metrics_history.append(metrics)
        
        # Evaluate
        if (episode + 1) % args.eval_freq == 0:
            eval_metrics = evaluate_policy(controller, env)
            print(f"\nEpisode {episode + 1}: Reward={eval_metrics['mean_reward']:.2f}, "
                  f"Energy={eval_metrics['mean_energy']:.2f}mWh, "
                  f"Battery={eval_metrics['mean_final_battery']:.1%}")
        
        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            controller.save(os.path.join(output_dir, f"checkpoint_{episode + 1}.pt"))
    
    # Save final model and metrics
    controller.save(os.path.join(output_dir, "final_model.pt"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {output_dir}")
    return metrics_history


def train_federated(args):
    """Train with federated learning across multiple devices."""
    print("=" * 60)
    print("FMADRL Federated Training")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Create multi-device environment
    multi_env = MultiDeviceEnvironment(
        n_devices=args.n_devices,
        episode_length=args.episode_length
    )
    
    # Get observation dimension from first device
    sample_obs = multi_env.devices[0].reset()
    obs_dim = multi_env.devices[0].observation_dim
    n_sensors = multi_env.devices[0].n_sensors
    
    # Create federated aggregator
    aggregator = FederatedAggregator(
        aggregation_method="fedavg",
        differential_privacy=args.dp,
        dp_epsilon=args.dp_epsilon
    )
    
    # Create controllers for each device
    controllers = [
        MultiAgentController(
            observation_dim=obs_dim,
            n_sensors=n_sensors,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=args.device
        )
        for _ in range(args.n_devices)
    ]
    
    # Initialize global model
    initial_params = controllers[0].get_model_parameters()
    aggregator.initialize_global_model(initial_params)
    
    # Create federated clients
    clients = [
        FederatedClient(
            client_id=i,
            local_epochs=args.local_epochs
        )
        for i in range(args.n_devices)
    ]
    
    # Training metrics
    round_metrics = []
    
    print(f"\nStarting federated training with {args.n_devices} devices...")
    print(f"Federated rounds: {args.fed_rounds}")
    print(f"Client fraction: {args.client_fraction}")
    print(f"Differential privacy: {'Enabled' if args.dp else 'Disabled'}")
    
    for fed_round in tqdm(range(args.fed_rounds), desc="Federated Rounds"):
        # Select participating clients
        n_selected = max(1, int(args.n_devices * args.client_fraction))
        selected_clients = np.random.choice(
            args.n_devices, size=n_selected, replace=False
        )
        
        # Distribute global model
        global_params = aggregator.get_global_model()
        
        # Local training on each selected client
        client_updates = []
        client_sample_counts = []
        client_rewards = []
        
        for client_idx in selected_clients:
            client = clients[client_idx]
            controller = controllers[client_idx]
            env = multi_env.devices[client_idx]
            
            # Receive global model
            client.receive_global_model(global_params)
            controller.set_model_parameters(global_params)
            
            # Collect local data through episodes
            local_data = []
            episode_reward = 0
            
            obs = env.reset()
            flat_obs = env.get_flat_observation()
            
            for _ in range(args.episode_length // 10):  # Shorter local episodes
                sensor_actions = controller.get_sensor_actions(flat_obs)
                next_obs, reward, done, info = env.step(sensor_actions)
                next_flat_obs = env.get_flat_observation()
                
                local_data.append((flat_obs, sensor_actions.flatten(), reward, next_flat_obs, done))
                
                flat_obs = next_flat_obs
                episode_reward += reward
                
                if done:
                    break
            
            # Local update
            updated_params = client.local_update(local_data, controller)
            
            client_updates.append(updated_params)
            client_sample_counts.append(len(local_data))
            client_rewards.append(episode_reward)
        
        # Aggregate updates
        if client_updates:
            aggregator.aggregate(
                client_updates,
                client_sample_counts=client_sample_counts
            )
        
        # Log round metrics
        metrics = {
            "round": fed_round,
            "n_clients": len(selected_clients),
            "mean_reward": np.mean(client_rewards),
            "std_reward": np.std(client_rewards),
            "total_samples": sum(client_sample_counts),
        }
        round_metrics.append(metrics)
        
        # Evaluate
        if (fed_round + 1) % args.eval_freq == 0:
            # Evaluate on first device
            eval_controller = controllers[0]
            eval_controller.set_model_parameters(aggregator.get_global_model())
            eval_metrics = evaluate_policy(eval_controller, multi_env.devices[0])
            
            print(f"\nRound {fed_round + 1}: "
                  f"Mean Reward={metrics['mean_reward']:.2f}, "
                  f"Eval Reward={eval_metrics['mean_reward']:.2f}, "
                  f"Energy={eval_metrics['mean_energy']:.2f}mWh")
    
    # Save final global model
    final_controller = controllers[0]
    final_controller.set_model_parameters(aggregator.get_global_model())
    final_controller.save(os.path.join(output_dir, "global_model.pt"))
    
    with open(os.path.join(output_dir, "round_metrics.json"), "w") as f:
        json.dump(round_metrics, f, indent=2)
    
    print(f"\nFederated training complete! Results saved to {output_dir}")
    return round_metrics


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    if args.n_devices == 1:
        # Single device training
        train_single_device(args)
    else:
        # Federated training
        train_federated(args)


if __name__ == "__main__":
    main()

