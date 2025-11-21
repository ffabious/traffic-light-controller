# Traffic Light Controller Environment Guide

This repository contains two Gymnasium-compatible environments for traffic light control:

1. **Single Intersection** (`TrafficIntersectionEnv`)
2. **Multi-Intersection** (`MultiIntersectionEnv`)

This guide explains how to interact with these environments, interpret states, and control the traffic lights.

## 1. Single Intersection Environment

### Initialization

```python
from environment.traffic_env import TrafficIntersectionEnv

# Initialize with custom parameters
env = TrafficIntersectionEnv(
    min_green_time=5,      # Minimum steps a phase must stay active
    yellow_time=2,         # Duration of yellow light (all red)
    switching_penalty=10.0 # Reward penalty for switching phases
)
```

### Observation Space (State)

The observation is a numpy array of shape `(5,)` containing:
| Index | Description | Range |
|-------|-------------|-------|
| 0 | Queue Length (North Lane) | [0, max_cars] |
| 1 | Queue Length (South Lane) | [0, max_cars] |
| 2 | Queue Length (East Lane) | [0, max_cars] |
| 3 | Queue Length (West Lane) | [0, max_cars] |
| 4 | Current Phase | 0 or 1 |

**Phases:**

- `0`: **North-South Green** (East-West Red)
- `1`: **East-West Green** (North-South Red)

### Action Space

The action is a discrete value:

- `0`: **Keep Phase** (Continue current green light)
- `1`: **Switch Phase** (Initiate switch to the other phase)

_Note: If you request a switch (`1`) before `min_green_time` has passed, the action is ignored and the phase remains unchanged._

### Rewards

The reward is designed to minimize delay:
$$ R_t = - (\sum_{lane} \text{QueueLength}_{lane}) - \text{SwitchPenalty} $$

- **Queue Penalty:** Negative sum of all cars waiting in all lanes.
- **Switch Penalty:** Applied only on the step when a switch actually occurs (to discourage rapid flickering).

---

## 2. Multi-Intersection Environment

### Initialization

```python
from environment.multi_intersection_env import MultiIntersectionEnv

env = MultiIntersectionEnv(
    min_green_time=5,
    yellow_time=2,
    switching_penalty=10.0
)
```

### Topology

Two intersections connected in series:

- **Intersection 1 (Left):** Feeds traffic East into Intersection 2.
- **Intersection 2 (Right):** Feeds traffic West into Intersection 1.

### Observation Space (State)

The observation is a numpy array of shape `(10,)` containing:

| Index | Intersection | Description                      |
| ----- | ------------ | -------------------------------- |
| 0-3   | **1**        | Queues: North, South, East, West |
| 4-7   | **2**        | Queues: North, South, East, West |
| 8     | **1**        | Current Phase (0=NS, 1=EW)       |
| 9     | **2**        | Current Phase (0=NS, 1=EW)       |

### Action Space

The action is a `MultiDiscrete([2, 2])` vector: `[action_int_1, action_int_2]`

- `action[0]`: Control for Intersection 1 (0=Keep, 1=Switch)
- `action[1]`: Control for Intersection 2 (0=Keep, 1=Switch)

### Rewards

The reward is the sum of rewards from both intersections:
$$ R\_{total} = R\_{int1} + R\_{int2} $$

---

## 3. Interaction Loop Example

Here is a standard interaction loop compatible with both environments:

```python
import gymnasium as gym
from environment.traffic_env import TrafficIntersectionEnv

env = TrafficIntersectionEnv()
obs, info = env.reset()

done = False
while not done:
    # 1. Select an action (Random agent example)
    action = env.action_space.sample()

    # 2. Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # 3. Access Info
    print(f"Throughput: {info['throughput']}, Wait Time: {info['total_wait']}")

    # 4. Check termination
    done = terminated or truncated

env.close()
```

## 4. Key Mechanics

### Yellow Light

When a switch is successfully triggered:

1.  The environment enters a **Yellow Phase** for `yellow_time` steps.
2.  During Yellow, **ALL** lights are red (no traffic flows).
3.  After `yellow_time` expires, the new phase becomes active (Green).

### Vehicle Flow

- **Arrival:** Vehicles arrive stochastically based on `arrival_rate` (defined in `Lane`).
- **Discharge:** If a lane is Green, 1 vehicle departs per step (if queue > 0).
- **Linking (Multi-Env):** Vehicles leaving Int 1 (Eastbound) are added to Int 2's West queue immediately.
