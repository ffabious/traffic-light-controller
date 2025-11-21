import gymnasium as gym
from gymnasium import spaces
import numpy as np
from lane import Lane

class TrafficIntersectionEnv(gym.Env):
    """
    A single intersection environment.
    Phase 0: North-South Green
    Phase 1: East-West Green
    """
    def __init__(self):
        super(TrafficIntersectionEnv, self).__init__()

        # Action: 0 = Set Phase NS, 1 = Set Phase EW
        self.action_space = spaces.Discrete(2)

        # Observation: [Q_North, Q_South, Q_East, Q_West, Current_Phase]
        # We assume a max queue length for normalization (e.g., 20 cars)
        self.max_cars = 50
        self.observation_space = spaces.Box(
            low=0, 
            high=self.max_cars, 
            shape=(5,), 
            dtype=np.float32
        )

        # Lanes: 0:N, 1:S, 2:E, 3:W
        self.lanes = [Lane(i) for i in range(4)]
        self.current_phase = 0 # Start with NS Green
        self.current_time = 0
        self.max_steps = 1000 # Episode duration

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.current_phase = 0
        for lane in self.lanes:
            lane.reset()
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_time += 1
        
        # Apply Action (Switch light)
        # Add logic here if you want a "yellow" phase or switching delay
        self.current_phase = action

        # Phase 0 (NS) Green: Lanes 0 and 1 flow
        # Phase 1 (EW) Green: Lanes 2 and 3 flow
        ns_green = (self.current_phase == 0)
        ew_green = (self.current_phase == 1)

        lane_queues = []
        total_step_wait = 0

        # Update all lanes
        # Lanes 0,1 are NS. Lanes 2,3 are EW.
        for i, lane in enumerate(self.lanes):
            is_green = False
            if i in [0, 1] and ns_green: is_green = True
            if i in [2, 3] and ew_green: is_green = True
            
            q_len = lane.step(is_green, self.current_time)
            lane_queues.append(q_len)
            total_step_wait += q_len

        # Reward: Negative total waiting time (Minimize delay)
        # You can add penalties for frequent switching if needed
        reward = -1.0 * total_step_wait

        # Check termination
        terminated = self.current_time >= self.max_steps
        truncated = False

        info = {
            "throughput": 0, # Needs tracking in Lane class to be accurate
            "total_wait": total_step_wait
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Return state vector
        obs = [l.queue for l in self.lanes] # Note: this returns deques, need len
        obs = [len(l.queue) for l in self.lanes]
        obs.append(self.current_phase)
        return np.array(obs, dtype=np.float32)

    def render(self):
        # Simple text render
        obs = self._get_obs()
        print(f"Time: {self.current_time} | Phase: {'NS' if obs[4]==0 else 'EW'}")
        print(f"NS Queues: {obs[0]}, {obs[1]} | EW Queues: {obs[2]}, {obs[3]}")
