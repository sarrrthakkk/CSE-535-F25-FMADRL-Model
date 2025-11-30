"""
Federated Learning for Privacy-Preserving Collaborative Optimization

Implements federated learning components for the FMADRL framework.
Enables multiple devices to collaboratively learn without sharing raw data.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy


class FederatedAggregator:
    """
    Server-side federated learning aggregator.
    Implements FedAvg and weighted averaging strategies.
    """
    
    def __init__(
        self,
        aggregation_method: str = "fedavg",
        differential_privacy: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        clip_norm: float = 1.0
    ):
        """
        Initialize federated aggregator.
        
        Args:
            aggregation_method: 'fedavg', 'weighted', or 'median'
            differential_privacy: Whether to use differential privacy
            dp_epsilon: Privacy budget epsilon
            dp_delta: Privacy budget delta
            clip_norm: Gradient clipping norm for DP
        """
        self.aggregation_method = aggregation_method
        self.differential_privacy = differential_privacy
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.clip_norm = clip_norm
        
        # Track global model
        self.global_model_params: Optional[Dict[str, torch.Tensor]] = None
        
        # Statistics
        self.round_number = 0
        self.participating_clients = []
        
    def initialize_global_model(self, initial_params: Dict[str, torch.Tensor]):
        """Initialize global model with initial parameters."""
        self.global_model_params = {
            k: v.clone() for k, v in initial_params.items()
        }
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        client_sample_counts: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: List of client model parameters
            client_weights: Optional weights for each client
            client_sample_counts: Number of samples per client
            
        Returns:
            Aggregated global model parameters
        """
        if len(client_updates) == 0:
            return self.global_model_params
        
        # Compute weights
        if client_weights is None:
            if client_sample_counts is not None:
                total_samples = sum(client_sample_counts)
                client_weights = [n / total_samples for n in client_sample_counts]
            else:
                client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Normalize weights
        weight_sum = sum(client_weights)
        client_weights = [w / weight_sum for w in client_weights]
        
        # Apply differential privacy if enabled
        if self.differential_privacy:
            client_updates = self._apply_differential_privacy(client_updates)
        
        # Aggregate based on method
        if self.aggregation_method == "fedavg":
            aggregated = self._fedavg(client_updates, client_weights)
        elif self.aggregation_method == "weighted":
            aggregated = self._weighted_average(client_updates, client_weights)
        elif self.aggregation_method == "median":
            aggregated = self._median_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        self.global_model_params = aggregated
        self.round_number += 1
        self.participating_clients.append(len(client_updates))
        
        return aggregated
    
    def _fedavg(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging (FedAvg) aggregation.
        McMahan et al., 2017
        """
        aggregated = {}
        
        # Get all parameter keys
        keys = client_updates[0].keys()
        
        for key in keys:
            # Weighted average of parameters
            weighted_sum = None
            for client_params, weight in zip(client_updates, weights):
                if weighted_sum is None:
                    weighted_sum = weight * client_params[key].clone()
                else:
                    weighted_sum += weight * client_params[key]
            
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def _weighted_average(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Weighted average (same as FedAvg but explicit)."""
        return self._fedavg(client_updates, weights)
    
    def _median_aggregation(
        self,
        client_updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise median aggregation.
        More robust to outliers/Byzantine clients.
        """
        aggregated = {}
        keys = client_updates[0].keys()
        
        for key in keys:
            # Stack all client parameters
            stacked = torch.stack([c[key] for c in client_updates])
            # Take median across clients
            aggregated[key] = torch.median(stacked, dim=0)[0]
        
        return aggregated
    
    def _apply_differential_privacy(
        self,
        client_updates: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply differential privacy through gradient clipping and noise addition.
        """
        processed_updates = []
        
        for update in client_updates:
            processed = {}
            
            for key, param in update.items():
                # Compute L2 norm
                param_norm = torch.norm(param.float())
                
                # Clip if necessary
                if param_norm > self.clip_norm:
                    param = param * (self.clip_norm / param_norm)
                
                # Add Gaussian noise
                noise_scale = self.clip_norm * np.sqrt(2 * np.log(1.25 / self.dp_delta)) / self.dp_epsilon
                noise = torch.normal(0, noise_scale, size=param.shape)
                
                processed[key] = param + noise
            
            processed_updates.append(processed)
        
        return processed_updates
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return {k: v.clone() for k, v in self.global_model_params.items()}
    
    def get_statistics(self) -> Dict:
        """Get aggregation statistics."""
        return {
            "round": self.round_number,
            "avg_clients_per_round": np.mean(self.participating_clients) if self.participating_clients else 0,
            "total_clients_served": sum(self.participating_clients),
        }


class FederatedClient:
    """
    Client-side federated learning component.
    Manages local training and communication with server.
    """
    
    def __init__(
        self,
        client_id: int,
        local_epochs: int = 5,
        local_batch_size: int = 32
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        
        # Track local model and data
        self.local_params: Optional[Dict[str, torch.Tensor]] = None
        self.sample_count = 0
        
        # Personalization layer (meta-learning)
        self.personalization_params: Optional[Dict[str, torch.Tensor]] = None
        self.personalization_lr = 0.01
        
    def receive_global_model(self, global_params: Dict[str, torch.Tensor]):
        """Receive global model from server."""
        self.local_params = {k: v.clone() for k, v in global_params.items()}
    
    def local_update(
        self,
        local_data: List[Tuple],
        controller: 'MultiAgentController'  # Type hint for the controller
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local training on client data.
        
        Args:
            local_data: List of (state, action, reward, next_state, done) tuples
            controller: The multi-agent controller to update
            
        Returns:
            Updated local model parameters
        """
        # Set controller parameters to global model
        controller.set_model_parameters(self.local_params)
        
        # Add local data to replay buffer
        for transition in local_data:
            controller.store_transition(*transition)
        
        self.sample_count = len(local_data)
        
        # Local training epochs
        metrics_history = []
        for epoch in range(self.local_epochs):
            if len(controller.replay_buffer) >= self.local_batch_size:
                metrics = controller.update(batch_size=self.local_batch_size)
                metrics_history.append(metrics)
        
        # Get updated parameters
        self.local_params = controller.get_model_parameters()
        
        return self.local_params
    
    def compute_gradient_update(
        self,
        global_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient update (difference from global model).
        Used for gradient-based aggregation methods.
        """
        if self.local_params is None:
            return {}
        
        gradients = {}
        for key in global_params.keys():
            if key in self.local_params:
                gradients[key] = self.local_params[key] - global_params[key]
        
        return gradients
    
    def personalize(
        self,
        personal_data: List[Tuple],
        controller: 'MultiAgentController',
        n_steps: int = 10
    ):
        """
        Personalize the global model using local data.
        Implements meta-learning style personalization.
        """
        # Start from global model
        controller.set_model_parameters(self.local_params)
        
        # Fine-tune on personal data
        for transition in personal_data:
            controller.store_transition(*transition)
        
        # Quick adaptation steps
        for _ in range(n_steps):
            if len(controller.replay_buffer) >= 16:
                controller.update(batch_size=16)
        
        # Store personalized parameters
        self.personalization_params = controller.get_model_parameters()


class FederatedTrainer:
    """
    Coordinates federated training across multiple clients.
    """
    
    def __init__(
        self,
        n_clients: int,
        aggregator: FederatedAggregator,
        client_fraction: float = 0.5,
        rounds: int = 100
    ):
        """
        Initialize federated trainer.
        
        Args:
            n_clients: Total number of clients
            aggregator: Server-side aggregator
            client_fraction: Fraction of clients to sample per round
            rounds: Number of training rounds
        """
        self.n_clients = n_clients
        self.aggregator = aggregator
        self.client_fraction = client_fraction
        self.rounds = rounds
        
        # Create clients
        self.clients = [
            FederatedClient(client_id=i)
            for i in range(n_clients)
        ]
        
        # Track training progress
        self.round_metrics = []
        
    def select_clients(self) -> List[int]:
        """Select subset of clients for this round."""
        n_selected = max(1, int(self.n_clients * self.client_fraction))
        return np.random.choice(
            self.n_clients, 
            size=n_selected, 
            replace=False
        ).tolist()
    
    def train_round(
        self,
        client_data: Dict[int, List[Tuple]],
        controllers: Dict[int, 'MultiAgentController']
    ) -> Dict:
        """
        Execute one round of federated training.
        
        Args:
            client_data: Dictionary mapping client_id to local data
            controllers: Dictionary mapping client_id to controller
            
        Returns:
            Round metrics
        """
        # Select participating clients
        selected_clients = self.select_clients()
        
        # Distribute global model to clients
        global_params = self.aggregator.get_global_model()
        
        client_updates = []
        client_sample_counts = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Receive global model
            client.receive_global_model(global_params)
            
            # Local training
            if client_id in client_data and client_id in controllers:
                local_params = client.local_update(
                    client_data[client_id],
                    controllers[client_id]
                )
                client_updates.append(local_params)
                client_sample_counts.append(client.sample_count)
        
        # Aggregate updates
        if client_updates:
            new_global = self.aggregator.aggregate(
                client_updates,
                client_sample_counts=client_sample_counts
            )
        
        metrics = {
            "round": self.aggregator.round_number,
            "n_clients": len(selected_clients),
            "total_samples": sum(client_sample_counts),
        }
        self.round_metrics.append(metrics)
        
        return metrics
    
    def run_training(
        self,
        data_generator,
        controller_factory
    ):
        """
        Run full federated training loop.
        
        Args:
            data_generator: Function that generates client data
            controller_factory: Function that creates a new controller
        """
        # Initialize global model
        template_controller = controller_factory()
        initial_params = template_controller.get_model_parameters()
        self.aggregator.initialize_global_model(initial_params)
        
        for round_num in range(self.rounds):
            # Generate data for this round
            client_data = data_generator()
            
            # Create controllers for participating clients
            controllers = {
                i: controller_factory()
                for i in range(self.n_clients)
            }
            
            # Train round
            metrics = self.train_round(client_data, controllers)
            
            if round_num % 10 == 0:
                print(f"Round {round_num}: {metrics}")


class TransferLearning:
    """
    Cross-device transfer learning component.
    Enables knowledge transfer across different device types.
    """
    
    def __init__(self):
        # Device profiles learned from different devices
        self.device_profiles: Dict[str, Dict] = {}
        
        # Shared feature layers (device-invariant)
        self.shared_layers = ["feature_net"]
        
        # Device-specific layers
        self.device_specific_layers = ["activation_head", "rate_head"]
    
    def extract_shared_params(
        self,
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract device-invariant parameters."""
        shared = {}
        for key, value in params.items():
            for layer in self.shared_layers:
                if layer in key:
                    shared[key] = value.clone()
                    break
        return shared
    
    def extract_device_specific_params(
        self,
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract device-specific parameters."""
        specific = {}
        for key, value in params.items():
            for layer in self.device_specific_layers:
                if layer in key:
                    specific[key] = value.clone()
                    break
        return specific
    
    def register_device(
        self,
        device_id: str,
        device_type: str,
        params: Dict[str, torch.Tensor]
    ):
        """Register a new device with its parameters."""
        self.device_profiles[device_id] = {
            "type": device_type,
            "shared": self.extract_shared_params(params),
            "specific": self.extract_device_specific_params(params),
        }
    
    def transfer_to_new_device(
        self,
        source_device_id: str,
        target_device_type: str
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer knowledge from source device to new device.
        
        Returns parameters initialized for the new device.
        """
        if source_device_id not in self.device_profiles:
            raise ValueError(f"Unknown source device: {source_device_id}")
        
        source = self.device_profiles[source_device_id]
        
        # Start with shared parameters
        new_params = source["shared"].copy()
        
        # Find similar device type for device-specific params
        similar_devices = [
            d for d in self.device_profiles.values()
            if d["type"] == target_device_type
        ]
        
        if similar_devices:
            # Use parameters from similar device
            new_params.update(similar_devices[0]["specific"])
        else:
            # Use source device-specific params as starting point
            new_params.update(source["specific"])
        
        return new_params
    
    def compute_similarity(
        self,
        device1_id: str,
        device2_id: str
    ) -> float:
        """Compute similarity between two device profiles."""
        if device1_id not in self.device_profiles or device2_id not in self.device_profiles:
            return 0.0
        
        d1 = self.device_profiles[device1_id]
        d2 = self.device_profiles[device2_id]
        
        # Compare shared parameters
        similarities = []
        for key in d1["shared"]:
            if key in d2["shared"]:
                p1 = d1["shared"][key].flatten()
                p2 = d2["shared"][key].flatten()
                cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
                similarities.append(cos_sim.item())
        
        return np.mean(similarities) if similarities else 0.0


# Import for type hints
try:
    import torch.nn.functional as F
except ImportError:
    pass


if __name__ == "__main__":
    # Test federated components
    print("Testing Federated Aggregator...")
    
    aggregator = FederatedAggregator(
        aggregation_method="fedavg",
        differential_privacy=False
    )
    
    # Create dummy client updates
    n_clients = 5
    param_dim = 100
    
    initial_params = {
        "layer1": torch.randn(param_dim, param_dim),
        "layer2": torch.randn(param_dim),
    }
    
    aggregator.initialize_global_model(initial_params)
    
    # Simulate client updates
    client_updates = []
    for i in range(n_clients):
        update = {
            "layer1": initial_params["layer1"] + torch.randn(param_dim, param_dim) * 0.1,
            "layer2": initial_params["layer2"] + torch.randn(param_dim) * 0.1,
        }
        client_updates.append(update)
    
    # Aggregate
    aggregated = aggregator.aggregate(client_updates)
    
    print(f"Initial layer1 mean: {initial_params['layer1'].mean():.4f}")
    print(f"Aggregated layer1 mean: {aggregated['layer1'].mean():.4f}")
    print(f"Statistics: {aggregator.get_statistics()}")
    
    print("\nTesting Transfer Learning...")
    transfer = TransferLearning()
    
    # Register devices
    for i in range(3):
        params = {
            "sensor_0_feature_net.0.weight": torch.randn(128, 20),
            "sensor_0_activation_head.0.weight": torch.randn(64, 128),
        }
        transfer.register_device(f"device_{i}", "smartphone", params)
    
    print(f"Registered devices: {list(transfer.device_profiles.keys())}")
    
    print("\nFederated learning components test passed!")

