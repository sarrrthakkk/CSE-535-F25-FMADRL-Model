"""
Multi-Agent Deep Reinforcement Learning for Sensor and Computation Coordination

Implements the multi-agent RL component of the FMADRL framework.
Each sensor and computation resource is modeled as an independent agent
that learns coordinated policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class SensorAgentNetwork(nn.Module):
    """
    Neural network for a single sensor agent.
    Outputs: activation probability and rate adjustment.
    """
    
    def __init__(
        self, 
        observation_dim: int, 
        hidden_dim: int = 128,
        sensor_idx: int = 0
    ):
        super().__init__()
        
        self.sensor_idx = sensor_idx
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Activation head (discrete: on/off)
        self.activation_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # on/off logits
        )
        
        # Rate adjustment head (continuous: -1 to 1)
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            activation_logits: Logits for on/off decision
            rate_adjustment: Continuous rate adjustment (-1 to 1)
        """
        features = self.feature_net(obs)
        activation_logits = self.activation_head(features)
        rate_adjustment = self.rate_head(features)
        
        return activation_logits, rate_adjustment
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """Get action for this sensor."""
        with torch.no_grad():
            activation_logits, rate_adj = self.forward(obs)
            
            if deterministic:
                activation = torch.argmax(activation_logits, dim=-1).item()
            else:
                probs = F.softmax(activation_logits, dim=-1)
                activation = torch.multinomial(probs, 1).item()
            
            rate = rate_adj.item()
            
        return activation, rate


class ComputeAgentNetwork(nn.Module):
    """
    Neural network for computation offloading decisions.
    Decides whether to execute locally, offload to edge, or cloud.
    """
    
    def __init__(
        self,
        observation_dim: int,
        max_tasks: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.max_tasks = max_tasks
        
        # Task-aware feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Offloading decision head (per-task)
        self.offload_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_tasks * 3),  # 3 options: local, edge, cloud
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning offloading logits."""
        features = self.feature_net(obs)
        logits = self.offload_head(features)
        return logits.view(-1, self.max_tasks, 3)
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        n_tasks: int,
        deterministic: bool = False
    ) -> np.ndarray:
        """Get offloading decisions for pending tasks."""
        with torch.no_grad():
            logits = self.forward(obs)
            
            if deterministic:
                actions = torch.argmax(logits[:, :n_tasks], dim=-1)
            else:
                probs = F.softmax(logits[:, :n_tasks], dim=-1)
                actions = torch.multinomial(probs.view(-1, 3), 1).view(-1, n_tasks)
            
        return actions.cpu().numpy().flatten()


class MultiAgentController:
    """
    Coordinates multiple sensor agents and compute agent.
    Implements centralized training with decentralized execution (CTDE).
    """
    
    def __init__(
        self,
        observation_dim: int,
        n_sensors: int,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu"
    ):
        self.observation_dim = observation_dim
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Create sensor agents
        self.sensor_agents = nn.ModuleList([
            SensorAgentNetwork(observation_dim, hidden_dim, i)
            for i in range(n_sensors)
        ]).to(device)
        
        # Create target networks for sensor agents
        self.sensor_agents_target = nn.ModuleList([
            SensorAgentNetwork(observation_dim, hidden_dim, i)
            for i in range(n_sensors)
        ]).to(device)
        
        # Copy weights to target
        for agent, target in zip(self.sensor_agents, self.sensor_agents_target):
            target.load_state_dict(agent.state_dict())
        
        # Create compute agent
        self.compute_agent = ComputeAgentNetwork(
            observation_dim, max_tasks=10, hidden_dim=hidden_dim
        ).to(device)
        
        self.compute_agent_target = ComputeAgentNetwork(
            observation_dim, max_tasks=10, hidden_dim=hidden_dim
        ).to(device)
        self.compute_agent_target.load_state_dict(self.compute_agent.state_dict())
        
        # Centralized critic for coordination
        self.critic = CentralizedCritic(
            observation_dim, n_sensors, hidden_dim
        ).to(device)
        
        self.critic_target = CentralizedCritic(
            observation_dim, n_sensors, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.sensor_optimizers = [
            optim.Adam(agent.parameters(), lr=lr)
            for agent in self.sensor_agents
        ]
        self.compute_optimizer = optim.Adam(self.compute_agent.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
    def get_sensor_actions(
        self, 
        obs: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """Get actions for all sensors."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        actions = []
        for agent in self.sensor_agents:
            activation, rate = agent.get_action(obs_tensor, deterministic)
            actions.append([activation, rate])
        
        return np.array(actions)
    
    def get_compute_actions(
        self, 
        obs: np.ndarray, 
        n_tasks: int,
        deterministic: bool = False
    ) -> np.ndarray:
        """Get offloading decisions."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.compute_agent.get_action(obs_tensor, n_tasks, deterministic)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action.flatten(), reward, next_state, done)
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Update all agents using sampled batch.
        
        Returns dict of training metrics.
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Reshape actions: [batch, n_sensors * 2]
        sensor_actions = actions[:, :self.n_sensors * 2].view(-1, self.n_sensors, 2)
        
        # Update critic
        with torch.no_grad():
            # Get target actions
            target_actions = []
            for i, agent in enumerate(self.sensor_agents_target):
                act_logits, rate = agent.forward(next_states)
                probs = F.softmax(act_logits, dim=-1)
                activation = (probs[:, 1] > 0.5).float().unsqueeze(1)
                target_actions.append(torch.cat([activation, rate], dim=1))
            target_actions = torch.stack(target_actions, dim=1)
            
            # Target Q-value
            target_q = self.critic_target(next_states, target_actions)
            target_value = rewards + self.gamma * (1 - dones) * target_q
        
        # Current Q-value
        current_q = self.critic(states, sensor_actions)
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update each sensor agent (actor)
        actor_losses = []
        for i, (agent, optimizer) in enumerate(zip(self.sensor_agents, self.sensor_optimizers)):
            # Get current agent's action
            act_logits, rate = agent.forward(states)
            probs = F.softmax(act_logits, dim=-1)
            activation = (probs[:, 1]).unsqueeze(1)  # Use soft probability
            
            # Construct action tensor for critic
            current_actions = sensor_actions.clone()
            current_actions[:, i] = torch.cat([activation, rate], dim=1)
            
            # Actor loss: maximize Q-value
            actor_loss = -self.critic(states, current_actions).mean()
            
            # Add entropy bonus for exploration
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            actor_loss = actor_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
            
            actor_losses.append(actor_loss.item())
        
        # Soft update targets
        self._soft_update()
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss_mean": np.mean(actor_losses),
        }
    
    def _soft_update(self):
        """Soft update target networks."""
        for agent, target in zip(self.sensor_agents, self.sensor_agents_target):
            for param, target_param in zip(agent.parameters(), target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all model parameters as a dictionary."""
        params = {}
        for i, agent in enumerate(self.sensor_agents):
            for name, param in agent.named_parameters():
                params[f"sensor_{i}_{name}"] = param.data.clone()
        
        for name, param in self.compute_agent.named_parameters():
            params[f"compute_{name}"] = param.data.clone()
        
        for name, param in self.critic.named_parameters():
            params[f"critic_{name}"] = param.data.clone()
            
        return params
    
    def set_model_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters from a dictionary."""
        for i, agent in enumerate(self.sensor_agents):
            for name, param in agent.named_parameters():
                key = f"sensor_{i}_{name}"
                if key in params:
                    param.data.copy_(params[key])
        
        for name, param in self.compute_agent.named_parameters():
            key = f"compute_{name}"
            if key in params:
                param.data.copy_(params[key])
        
        for name, param in self.critic.named_parameters():
            key = f"critic_{name}"
            if key in params:
                param.data.copy_(params[key])
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'sensor_agents': [a.state_dict() for a in self.sensor_agents],
            'compute_agent': self.compute_agent.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path)
        for i, agent in enumerate(self.sensor_agents):
            agent.load_state_dict(checkpoint['sensor_agents'][i])
        self.compute_agent.load_state_dict(checkpoint['compute_agent'])
        self.critic.load_state_dict(checkpoint['critic'])


class CentralizedCritic(nn.Module):
    """
    Centralized critic for multi-agent training.
    Takes observations and all agents' actions to estimate Q-value.
    """
    
    def __init__(
        self,
        observation_dim: int,
        n_sensors: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Input: observation + all sensor actions
        action_dim = n_sensors * 2  # activation + rate per sensor
        input_dim = observation_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-value.
        
        Args:
            obs: Observations [batch, obs_dim]
            actions: All sensor actions [batch, n_sensors, 2]
        """
        actions_flat = actions.view(obs.size(0), -1)
        x = torch.cat([obs, actions_flat], dim=1)
        return self.net(x)


if __name__ == "__main__":
    # Test the multi-agent controller
    obs_dim = 20
    n_sensors = 6
    
    controller = MultiAgentController(obs_dim, n_sensors)
    
    # Generate random observation
    obs = np.random.randn(obs_dim).astype(np.float32)
    
    # Get actions
    sensor_actions = controller.get_sensor_actions(obs)
    print("Sensor actions shape:", sensor_actions.shape)
    print("Sensor actions:\n", sensor_actions)
    
    compute_actions = controller.get_compute_actions(obs, n_tasks=3)
    print("Compute actions:", compute_actions)
    
    # Test training loop
    print("\nTesting training loop...")
    for i in range(100):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = controller.get_sensor_actions(obs)
        reward = np.random.randn()
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        done = i == 99
        
        controller.store_transition(obs, action, reward, next_obs, done)
    
    metrics = controller.update(batch_size=32)
    print("Training metrics:", metrics)
    
    print("\nMulti-agent controller test passed!")

