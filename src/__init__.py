"""
FMADRL: Federated Multi-Agent Deep Reinforcement Learning
for Mobile Energy Optimization

This package implements the framework described in:
"Federated Multi-Agent Deep Reinforcement Learning for Unified Energy 
Optimization in Mobile Sensing and Edge Computing"
"""

from .environment import (
    MobileDeviceEnvironment,
    MultiDeviceEnvironment,
    SensorType,
    SensorConfig,
    TaskConfig,
)

from .agents import (
    MultiAgentController,
    SensorAgentNetwork,
    ComputeAgentNetwork,
    ReplayBuffer,
)

from .federated import (
    FederatedAggregator,
    FederatedClient,
    TransferLearning,
)

__version__ = "0.1.0"
__author__ = "FMADRL Research Team"

