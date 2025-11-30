"""
Mobile Device Energy Optimization Environment

Simulates a mobile device with multiple sensors and computation tasks.
Implements the environment for multi-agent reinforcement learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    """Types of sensors available on mobile devices."""
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    MICROPHONE = "microphone"
    CAMERA = "camera"


@dataclass
class SensorConfig:
    """Configuration for a sensor."""
    name: str
    sensor_type: SensorType
    power_consumption: float  # mW when active
    sampling_rate: float  # Hz
    utility_weight: float  # Importance for context detection
    min_rate: float = 1.0  # Minimum sampling rate
    max_rate: float = 100.0  # Maximum sampling rate


@dataclass
class TaskConfig:
    """Configuration for a computation task."""
    name: str
    cpu_cycles: int  # Million cycles
    data_size: float  # MB
    deadline: float  # seconds
    priority: float  # 0-1


# Default sensor configurations based on research papers [b1, b4]
DEFAULT_SENSORS = [
    SensorConfig("GPS", SensorType.GPS, power_consumption=400.0, 
                 sampling_rate=1.0, utility_weight=0.9, min_rate=0.1, max_rate=10.0),
    SensorConfig("Accelerometer", SensorType.ACCELEROMETER, power_consumption=10.0,
                 sampling_rate=50.0, utility_weight=0.7, min_rate=10.0, max_rate=200.0),
    SensorConfig("Gyroscope", SensorType.GYROSCOPE, power_consumption=15.0,
                 sampling_rate=50.0, utility_weight=0.6, min_rate=10.0, max_rate=200.0),
    SensorConfig("WiFi", SensorType.WIFI, power_consumption=200.0,
                 sampling_rate=0.1, utility_weight=0.5, min_rate=0.01, max_rate=1.0),
    SensorConfig("Bluetooth", SensorType.BLUETOOTH, power_consumption=50.0,
                 sampling_rate=0.5, utility_weight=0.4, min_rate=0.1, max_rate=5.0),
    SensorConfig("Microphone", SensorType.MICROPHONE, power_consumption=100.0,
                 sampling_rate=1.0, utility_weight=0.5, min_rate=0.1, max_rate=10.0),
]


class UserActivityPattern:
    """Simulates user activity patterns for context-aware sensing."""
    
    PATTERNS = {
        "stationary": {"movement": 0.1, "location_change": 0.05, "app_usage": 0.3},
        "walking": {"movement": 0.7, "location_change": 0.3, "app_usage": 0.5},
        "driving": {"movement": 0.9, "location_change": 0.8, "app_usage": 0.2},
        "working": {"movement": 0.2, "location_change": 0.1, "app_usage": 0.8},
        "sleeping": {"movement": 0.05, "location_change": 0.01, "app_usage": 0.05},
    }
    
    def __init__(self, pattern_probs: Optional[Dict[str, float]] = None):
        self.patterns = list(self.PATTERNS.keys())
        self.probs = pattern_probs or {p: 1.0/len(self.patterns) for p in self.patterns}
        self.current_pattern = np.random.choice(self.patterns)
        self.time_in_pattern = 0
        self.pattern_duration = np.random.exponential(300)  # seconds
        
    def step(self, dt: float = 1.0) -> str:
        """Update activity pattern based on time."""
        self.time_in_pattern += dt
        if self.time_in_pattern >= self.pattern_duration:
            # Transition to new pattern
            probs = [self.probs[p] for p in self.patterns]
            probs = np.array(probs) / sum(probs)
            self.current_pattern = np.random.choice(self.patterns, p=probs)
            self.time_in_pattern = 0
            self.pattern_duration = np.random.exponential(300)
        return self.current_pattern
    
    def get_context(self) -> Dict[str, float]:
        """Get current context features."""
        return self.PATTERNS[self.current_pattern].copy()


class MobileDeviceEnvironment:
    """
    Simulates a mobile device environment for energy optimization.
    
    Based on insights from:
    - Sensors Power Hungry [b1]: Energy consumption patterns
    - EEMSS [b4]: Hierarchical sensor management
    - Battery-Aware MEC [b5]: Battery-level considerations
    """
    
    def __init__(
        self,
        sensors: Optional[List[SensorConfig]] = None,
        battery_capacity: float = 4000.0,  # mAh
        voltage: float = 3.7,  # V
        initial_battery: float = 1.0,  # Fraction
        time_step: float = 1.0,  # seconds
        episode_length: int = 3600,  # 1 hour in seconds
    ):
        self.sensors = sensors or DEFAULT_SENSORS
        self.n_sensors = len(self.sensors)
        self.battery_capacity = battery_capacity * voltage  # Convert to mWh
        self.initial_battery = initial_battery
        self.time_step = time_step
        self.episode_length = episode_length
        
        # State tracking
        self.battery_level = initial_battery
        self.sensor_states = np.zeros(self.n_sensors)  # 0=off, 1=on
        self.sensor_rates = np.array([s.sampling_rate for s in self.sensors])
        self.current_step = 0
        
        # User activity simulation
        self.user_activity = UserActivityPattern()
        
        # Computation tasks queue
        self.task_queue: List[TaskConfig] = []
        self.completed_tasks = 0
        
        # Network conditions (for offloading decisions)
        self.network_quality = 1.0  # 0-1
        self.network_type = "wifi"  # wifi, lte, none
        
        # Metrics tracking
        self.total_energy_consumed = 0.0
        self.total_utility = 0.0
        self.sensing_events = []
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to initial state."""
        self.battery_level = self.initial_battery
        self.sensor_states = np.zeros(self.n_sensors)
        self.sensor_rates = np.array([s.sampling_rate for s in self.sensors])
        self.current_step = 0
        self.user_activity = UserActivityPattern()
        self.task_queue = []
        self.completed_tasks = 0
        self.network_quality = np.random.uniform(0.5, 1.0)
        self.total_energy_consumed = 0.0
        self.total_utility = 0.0
        self.sensing_events = []
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation/state."""
        context = self.user_activity.get_context()
        
        obs = {
            # Battery state
            "battery_level": np.array([self.battery_level]),
            
            # Sensor states and rates
            "sensor_states": self.sensor_states.copy(),
            "sensor_rates": self.sensor_rates / 100.0,  # Normalize
            
            # Context features
            "movement": np.array([context["movement"]]),
            "location_change": np.array([context["location_change"]]),
            "app_usage": np.array([context["app_usage"]]),
            
            # Network state
            "network_quality": np.array([self.network_quality]),
            "network_type": np.array([1.0 if self.network_type == "wifi" else 0.5 if self.network_type == "lte" else 0.0]),
            
            # Time features
            "time_of_day": np.array([np.sin(2 * np.pi * self.current_step / 86400),
                                      np.cos(2 * np.pi * self.current_step / 86400)]),
            
            # Task queue info
            "pending_tasks": np.array([len(self.task_queue) / 10.0]),  # Normalize
        }
        
        return obs
    
    def get_flat_observation(self) -> np.ndarray:
        """Get flattened observation for simple RL agents."""
        obs = self._get_observation()
        return np.concatenate([v.flatten() for v in obs.values()])
    
    @property
    def observation_dim(self) -> int:
        """Dimension of flattened observation."""
        return len(self.get_flat_observation())
    
    def _compute_sensor_energy(self) -> float:
        """Compute energy consumed by active sensors in this time step."""
        energy = 0.0
        for i, sensor in enumerate(self.sensors):
            if self.sensor_states[i] > 0:
                # Energy = Power * Time, adjusted for sampling rate
                rate_factor = self.sensor_rates[i] / sensor.sampling_rate
                energy += sensor.power_consumption * rate_factor * self.time_step / 3600.0  # Convert to mWh
        return energy
    
    def _compute_sensing_utility(self) -> float:
        """
        Compute utility of current sensing configuration.
        Based on EEMSS [b4] approach: utility depends on context and active sensors.
        """
        context = self.user_activity.get_context()
        utility = 0.0
        
        for i, sensor in enumerate(self.sensors):
            if self.sensor_states[i] > 0:
                # Utility based on sensor importance and context relevance
                context_relevance = self._get_context_relevance(sensor.sensor_type, context)
                rate_utility = min(1.0, self.sensor_rates[i] / sensor.sampling_rate)
                utility += sensor.utility_weight * context_relevance * rate_utility
                
        return utility
    
    def _get_context_relevance(self, sensor_type: SensorType, context: Dict[str, float]) -> float:
        """Get relevance of a sensor type given current context."""
        relevance_map = {
            SensorType.GPS: context["location_change"],
            SensorType.ACCELEROMETER: context["movement"],
            SensorType.GYROSCOPE: context["movement"] * 0.8,
            SensorType.WIFI: context["app_usage"] * 0.7 + 0.3,
            SensorType.BLUETOOTH: 0.3,
            SensorType.MAGNETOMETER: context["movement"] * 0.5,
            SensorType.MICROPHONE: context["app_usage"] * 0.5,
            SensorType.CAMERA: context["app_usage"] * 0.3,
        }
        return relevance_map.get(sensor_type, 0.5)
    
    def _generate_task(self) -> Optional[TaskConfig]:
        """Randomly generate a computation task."""
        if np.random.random() < 0.1:  # 10% chance per step
            context = self.user_activity.get_context()
            app_usage = context["app_usage"]
            
            return TaskConfig(
                name=f"task_{self.current_step}",
                cpu_cycles=int(np.random.exponential(100) * (1 + app_usage)),
                data_size=np.random.exponential(1.0) * (1 + app_usage),
                deadline=np.random.uniform(1.0, 10.0),
                priority=np.random.random()
            )
        return None
    
    def _update_network(self):
        """Update network conditions."""
        # Simulate network quality fluctuations
        self.network_quality = np.clip(
            self.network_quality + np.random.normal(0, 0.1),
            0.1, 1.0
        )
        
        # Occasionally change network type
        if np.random.random() < 0.01:
            self.network_type = np.random.choice(["wifi", "lte", "none"], p=[0.6, 0.3, 0.1])
    
    def step(
        self, 
        sensor_actions: np.ndarray,  # Shape: (n_sensors, 2) - [on/off, rate_adjustment]
        compute_actions: Optional[np.ndarray] = None  # Offloading decisions
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            sensor_actions: Actions for each sensor [activate, rate_change]
            compute_actions: Actions for computation offloading
            
        Returns:
            observation, reward, done, info
        """
        # Process sensor actions
        for i in range(self.n_sensors):
            if sensor_actions.ndim == 2:
                activate = sensor_actions[i, 0] > 0.5
                rate_change = sensor_actions[i, 1]  # -1 to 1
            else:
                activate = sensor_actions[i] > 0.5
                rate_change = 0.0
                
            self.sensor_states[i] = 1.0 if activate else 0.0
            
            # Adjust sampling rate
            sensor = self.sensors[i]
            new_rate = self.sensor_rates[i] * (1 + rate_change * 0.5)
            self.sensor_rates[i] = np.clip(new_rate, sensor.min_rate, sensor.max_rate)
        
        # Update user activity
        self.user_activity.step(self.time_step)
        
        # Generate new tasks
        new_task = self._generate_task()
        if new_task:
            self.task_queue.append(new_task)
        
        # Process computation (simplified)
        if compute_actions is not None and len(self.task_queue) > 0:
            # Process tasks based on offloading decisions
            completed = []
            for i, task in enumerate(self.task_queue):
                if i < len(compute_actions):
                    if compute_actions[i] > 0.5:  # Offload
                        if self.network_quality > 0.3:
                            completed.append(i)
                    else:  # Local execution
                        completed.append(i)
            
            # Remove completed tasks
            self.task_queue = [t for i, t in enumerate(self.task_queue) if i not in completed]
            self.completed_tasks += len(completed)
        
        # Update network conditions
        self._update_network()
        
        # Compute energy consumption
        energy_consumed = self._compute_sensor_energy()
        self.total_energy_consumed += energy_consumed
        
        # Update battery
        battery_drain = energy_consumed / self.battery_capacity
        self.battery_level = max(0.0, self.battery_level - battery_drain)
        
        # Compute utility
        utility = self._compute_sensing_utility()
        self.total_utility += utility
        
        # Compute reward using battery-aware reward shaping [b5]
        reward = self._compute_reward(utility, energy_consumed)
        
        # Check termination
        self.current_step += 1
        done = self.current_step >= self.episode_length or self.battery_level <= 0.0
        
        # Get new observation
        obs = self._get_observation()
        
        info = {
            "energy_consumed": energy_consumed,
            "utility": utility,
            "battery_level": self.battery_level,
            "active_sensors": int(np.sum(self.sensor_states)),
            "completed_tasks": self.completed_tasks,
            "current_activity": self.user_activity.current_pattern,
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self, utility: float, energy: float) -> float:
        """
        Compute battery-aware reward.
        Based on battery-level weighting from [b5].
        
        R = ω(b) * U - (1 - ω(b)) * E
        
        where ω(b) decreases as battery decreases (more conservative)
        """
        # Battery-dependent weight (omega)
        # When battery is high, prioritize utility; when low, prioritize energy saving
        omega = self.battery_level ** 0.5  # Square root for smoother transition
        
        # Normalize energy cost (assuming max ~500mWh per step)
        normalized_energy = energy / 0.5
        
        # Reward computation
        reward = omega * utility - (1 - omega) * normalized_energy
        
        # Penalty for battery depletion
        if self.battery_level < 0.1:
            reward -= 1.0
        if self.battery_level <= 0.0:
            reward -= 10.0
            
        return reward
    
    def get_sensor_action_space(self) -> Tuple[int, int]:
        """Return action space dimensions for sensors."""
        # Each sensor: [on/off (discrete), rate_adjustment (continuous)]
        return self.n_sensors, 2
    
    def get_compute_action_space(self) -> int:
        """Return action space dimension for computation."""
        # Binary decision per task: offload or local
        return 10  # Max tasks to consider


class MultiDeviceEnvironment:
    """
    Wrapper for simulating multiple devices for federated learning.
    """
    
    def __init__(self, n_devices: int = 10, **env_kwargs):
        self.n_devices = n_devices
        self.devices = [MobileDeviceEnvironment(**env_kwargs) for _ in range(n_devices)]
        
        # Personalization: slightly different user patterns per device
        for i, device in enumerate(self.devices):
            # Vary user activity patterns
            patterns = list(UserActivityPattern.PATTERNS.keys())
            probs = {p: np.random.dirichlet(np.ones(len(patterns)))[j] 
                     for j, p in enumerate(patterns)}
            device.user_activity = UserActivityPattern(pattern_probs=probs)
            
            # Vary initial battery
            device.battery_level = np.random.uniform(0.5, 1.0)
    
    def reset_all(self) -> List[Dict[str, np.ndarray]]:
        """Reset all devices."""
        return [device.reset() for device in self.devices]
    
    def step_all(
        self, 
        actions: List[np.ndarray]
    ) -> List[Tuple[Dict[str, np.ndarray], float, bool, Dict]]:
        """Step all devices."""
        results = []
        for device, action in zip(self.devices, actions):
            results.append(device.step(action))
        return results


if __name__ == "__main__":
    # Test the environment
    env = MobileDeviceEnvironment()
    obs = env.reset()
    
    print("Observation keys:", obs.keys())
    print("Flat observation dim:", env.observation_dim)
    print("Sensor action space:", env.get_sensor_action_space())
    
    # Run a few random steps
    total_reward = 0
    for _ in range(100):
        # Random sensor actions
        actions = np.random.rand(env.n_sensors, 2)
        actions[:, 0] = (actions[:, 0] > 0.5).astype(float)
        actions[:, 1] = actions[:, 1] * 2 - 1  # Scale to [-1, 1]
        
        obs, reward, done, info = env.step(actions)
        total_reward += reward
        
        if done:
            break
    
    print(f"\nTest run completed:")
    print(f"  Steps: {env.current_step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final battery: {env.battery_level:.2%}")
    print(f"  Energy consumed: {env.total_energy_consumed:.2f} mWh")

