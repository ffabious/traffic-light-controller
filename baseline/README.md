# Regular Traffic Light Controller (Baseline)

A classical traffic light control system that serves as a baseline benchmark for RL experiments. This controller implements fixed-time signal control with cycle-based phase sequencing and offset coordination between two connected intersections.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Controller Classes](#controller-classes)
- [Configuration Parameters](#configuration-parameters)
- [Examples](#examples)
- [Testing](#testing)
- [How It Works](#how-it-works)

## Quick Start

Get started in 3 steps:

```python
from environment.multi_env import MultiIntersectionEnv
from baseline.regular_controller import FixedTimeController

# Step 1: Create the environment
env = MultiIntersectionEnv(min_green_time=5, yellow_time=2, switching_penalty=10.0)

# Step 2: Create the controller
controller = FixedTimeController(
    cycle_time=30,
    green_time_ns=12,
    green_time_ew=12,
    yellow_time=2,
    offset=None,  # Auto-calculate optimal offset
    travel_time=8
)

# Step 3: Run the simulation
obs, info = env.reset()
controller.reset()

for step in range(1000):
    # Get action from controller
    action = controller.get_action(env.current_time)
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

print(f"Simulation completed!")
print(f"Total reward: {reward}")
print(f"Total throughput: {info['throughput']}")
```

## Installation

No additional installation required! The baseline controller uses only standard Python libraries (numpy) and the existing environment.

Make sure you're in the project directory:
```bash
cd project/traffic-light-controller
```

## Basic Usage

### Minimal Example

```python
from environment.multi_env import MultiIntersectionEnv
from baseline.regular_controller import FixedTimeController

# Initialize
env = MultiIntersectionEnv()
controller = FixedTimeController()  # Uses default parameters

# Run
obs, _ = env.reset()
controller.reset()

for _ in range(100):
    action = controller.get_action(env.current_time)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### With Custom Parameters

```python
controller = FixedTimeController(
    cycle_time=40,          # Longer cycle for heavier traffic
    green_time_ns=18,       # Longer NS green phase
    green_time_ew=18,       # Longer EW green phase
    yellow_time=2,          # Yellow clearance time
    offset=10,              # Manual offset (or None for auto)
    travel_time=12          # Travel time between intersections
)
```

## Controller Classes

### `FixedTimeController` (Recommended)

**Best for**: Benchmarking and production use

- **Stateless**: Fast execution, no internal state tracking
- **Time-based**: Determines actions purely from current time
- **Lightweight**: Minimal overhead

```python
from baseline.regular_controller import FixedTimeController

controller = FixedTimeController(
    cycle_time=30,
    green_time_ns=12,
    green_time_ew=12,
    yellow_time=2,
    offset=None,
    travel_time=8
)

# Get action for current time step
action = controller.get_action(current_time)  # Returns (action_int1, action_int2)

# Get current phases
phases = controller.get_current_phases(current_time)  # Returns (phase1, phase2)
```

### `RegularTrafficLightController`

**Best for**: Debugging and monitoring

- **Stateful**: Maintains internal phase state
- **Trackable**: Can query current state and phase information
- **Debuggable**: Useful for understanding controller behavior

```python
from baseline.regular_controller import RegularTrafficLightController

controller = RegularTrafficLightController(
    cycle_time=30,
    green_time_ns=12,
    green_time_ew=12,
    yellow_time=2,
    offset=7,
    travel_time=8
)

controller.reset()

# Get action
action = controller.get_action(current_time)

# Get current phases
phases = controller.get_current_phases()  # (phase1, phase2)

# Get detailed state information
info = controller.get_info()
print(info)
# {
#     'cycle_time': 28,
#     'offset': 7,
#     'phase_state_1': 0,
#     'phase_state_2': 0,
#     ...
# }
```

## Configuration Parameters

### Cycle Time

The total duration of one complete cycle: `NS green + Yellow + EW green + Yellow`

**Default**: Auto-calculated from phase durations

**Example**:
- `green_time_ns=12`, `green_time_ew=12`, `yellow_time=2`
- Cycle time = `12 + 2 + 12 + 2 = 28` steps

**Note**: If you specify `cycle_time` that doesn't match the sum of phases, it will be automatically corrected with a warning.

### Phase Durations

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `green_time_ns` | North-South green phase duration | 12 | 10-20 steps |
| `green_time_ew` | East-West green phase duration | 12 | 10-20 steps |
| `yellow_time` | Yellow/all-red clearance interval | 2 | 2-4 steps |

### Offset

The time difference between when the two intersections start their cycles. This creates coordination and enables "green wave" effects.

**Options**:
- `offset=None` (default): Auto-calculate using `min(cycle_time/2, travel_time-1)`
- `offset=0`: Simultaneous plan (both intersections change together)
- `offset=7`: Manual offset (e.g., 7 steps after intersection 1)

**Recommendation**: Use `offset=None` with appropriate `travel_time` for optimal coordination.

### Travel Time

Estimated travel time between intersections (used for auto-calculating offset).

**Default**: 8 steps

**Rule of thumb**: Set offset to `travel_time - 1` for optimal platoon coordination.

## Examples

### Example 1: Simple Evaluation

```python
from environment.multi_env import MultiIntersectionEnv
from baseline.regular_controller import FixedTimeController
import numpy as np

env = MultiIntersectionEnv(min_green_time=5, yellow_time=2)
controller = FixedTimeController()

# Run multiple episodes
num_episodes = 10
results = []

for episode in range(num_episodes):
    obs, info = env.reset()
    controller.reset()
    
    episode_reward = 0
    episode_throughput = 0
    
    for step in range(1000):
        action = controller.get_action(env.current_time)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_throughput += info['throughput']
        
        if terminated or truncated:
            break
    
    results.append({
        'reward': episode_reward,
        'throughput': episode_throughput
    })

# Report statistics
print(f"Average Reward: {np.mean([r['reward'] for r in results]):.2f}")
print(f"Average Throughput: {np.mean([r['throughput'] for r in results]):.0f}")
print(f"Std Dev Reward: {np.std([r['reward'] for r in results]):.2f}")
```

### Example 2: Comparing Different Configurations

```python
from environment.multi_env import MultiIntersectionEnv
from baseline.regular_controller import FixedTimeController

configs = [
    {"name": "Short Cycle", "cycle_time": 20, "green_time_ns": 8, "green_time_ew": 8},
    {"name": "Long Cycle", "cycle_time": 40, "green_time_ns": 18, "green_time_ew": 18},
    {"name": "Asymmetric", "cycle_time": 30, "green_time_ns": 18, "green_time_ew": 10},
]

env = MultiIntersectionEnv()

for config in configs:
    controller = FixedTimeController(
        cycle_time=config["cycle_time"],
        green_time_ns=config["green_time_ns"],
        green_time_ew=config["green_time_ew"],
        yellow_time=2
    )
    
    obs, _ = env.reset()
    controller.reset()
    
    total_reward = 0
    for step in range(500):
        action = controller.get_action(env.current_time)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"{config['name']}: Reward = {total_reward:.2f}")
```

### Example 3: Monitoring Controller State

```python
from baseline.regular_controller import RegularTrafficLightController

controller = RegularTrafficLightController()
controller.reset()

for t in range(100):
    action = controller.get_action(t)
    phases = controller.get_current_phases()
    info = controller.get_info()
    
    if t % 10 == 0:
        print(f"Time {t:3d}: "
              f"Action={action}, "
              f"Phases=({phases[0]}, {phases[1]}), "
              f"State1={info['phase_state_1']}, "
              f"State2={info['phase_state_2']}")
```

### Example 4: Integration with RL Training Loop

```python
from environment.multi_env import MultiIntersectionEnv
from baseline.regular_controller import FixedTimeController

# Create baseline controller for comparison
baseline_controller = FixedTimeController()

# Your RL agent (example)
class RLAgent:
    def get_action(self, obs):
        # Your RL policy here
        return [0, 0]  # Placeholder

rl_agent = RLAgent()
env = MultiIntersectionEnv()

# Compare baseline vs RL agent
for agent_name, agent in [("Baseline", baseline_controller), ("RL", rl_agent)]:
    obs, _ = env.reset()
    if hasattr(agent, 'reset'):
        agent.reset()
    
    total_reward = 0
    for step in range(1000):
        if isinstance(agent, FixedTimeController):
            action = agent.get_action(env.current_time)
        else:
            action = agent.get_action(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"{agent_name} Total Reward: {total_reward:.2f}")
```

## Testing

### Validate Controller Logic

Test the controller logic without running the full environment:

```bash
cd project/traffic-light-controller
python baseline/validate_controller.py
```

This will verify:
- Cycle timing correctness
- Phase sequence logic
- Offset coordination
- State tracking (for RegularTrafficLightController)

### Full Integration Test

Test with the actual environment:

```bash
python baseline/test_baseline.py
```

**Note**: Requires `gymnasium` to be installed. Install dependencies with:
```bash
pip install -r requirements.txt
```

## How It Works

### Phase Sequence

The controller follows a fixed cycle:

1. **NS Green** → Vehicles flow North-South (duration: `green_time_ns`)
2. **Yellow/All-Red** → Clearance interval (duration: `yellow_time`)
3. **EW Green** → Vehicles flow East-West (duration: `green_time_ew`)
4. **Yellow/All-Red** → Clearance interval (duration: `yellow_time`)
5. **Repeat** from step 1

### Coordination

The controller coordinates two intersections using an **offset**:

- **Intersection 1** starts its cycle at time `t = 0`
- **Intersection 2** starts its cycle at time `t = offset`

This creates a "green wave" effect where vehicles can progress through both intersections without stopping, assuming the offset matches the travel time between intersections.

### Action Generation

The controller determines when to switch phases based on the current time:

```python
# For intersection 1
time_in_cycle_1 = current_time % cycle_time

# For intersection 2 (with offset)
time_in_cycle_2 = (current_time - offset) % cycle_time

# Switch at phase boundaries:
# - Start of cycle (t=0)
# - End of NS green (t=green_time_ns)
# - End of yellow after NS (t=green_time_ns + yellow_time)
# - End of EW green (t=green_time_ns + yellow_time + green_time_ew)
# - End of yellow after EW (t=cycle_time)
```

### Conflict Monitoring

The environment ensures no conflicting movements receive green simultaneously. The controller doesn't need to handle this - it's enforced by the environment's `min_green_time` constraint.

## Tips and Best Practices

1. **Use `FixedTimeController` for benchmarking**: It's faster and has no overhead
2. **Auto-calculate offset**: Set `offset=None` and provide realistic `travel_time`
3. **Match cycle time to phase durations**: Let the controller auto-correct, or calculate manually
4. **Test different configurations**: Experiment with cycle times and phase durations for your traffic patterns
5. **Monitor with `RegularTrafficLightController`**: Use the stateful version when debugging

## Troubleshooting

### Warning: cycle_time doesn't match calculated cycle

**Solution**: This is normal! The controller automatically corrects the cycle time. Either:
- Let it auto-correct (recommended)
- Calculate manually: `cycle_time = green_time_ns + yellow_time + green_time_ew + yellow_time`

### Offset seems wrong

**Check**: The offset is the time difference between cycle starts, not between first switches. Intersection 2's first switch may occur before its cycle officially starts if it's in the middle of a phase.

### Controller not switching phases

**Check**: 
- Ensure `min_green_time` in environment is less than your green phase durations
- Verify the controller is being called with `get_action(env.current_time)`
- Check that actions are being passed to `env.step(action)`

## References

- Traffic signal coordination principles
- Single-alternating coordination plan
- Cycle-based fixed-time control
- See `ENV_GUIDE.md` for environment details
