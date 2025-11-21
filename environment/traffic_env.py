import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.lane import Lane

class TrafficIntersectionEnv(gym.Env):
    """
    A single intersection environment.
    Phase 0: North-South Green
    Phase 1: East-West Green
    """
    def __init__(self, min_green_time=5, yellow_time=2, switching_penalty=10.0):
        super(TrafficIntersectionEnv, self).__init__()

        # Action: 0 = Keep current phase, 1 = Switch phase
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
        
        # Switching constraints
        self.min_green_time = min_green_time
        self.yellow_time = yellow_time
        self.switching_penalty = switching_penalty
        self.last_switch_time = -self.min_green_time  # Allow immediate first switch
        self.yellow_timer = 0  # Tracks remaining yellow light duration

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.current_phase = 0
        self.last_switch_time = -self.min_green_time
        self.yellow_timer = 0
        for lane in self.lanes:
            lane.reset()
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_time += 1
        
        # Handle yellow light timer
        if self.yellow_timer > 0:
            self.yellow_timer -= 1
            # All lights are red during yellow phase
            phase_is_active = False
        else:
            # Action interpretation: 0 = Keep current phase, 1 = Switch phase
            switch_requested = (action == 1)
            time_since_switch = self.current_time - self.last_switch_time
            
            # Check if we can perform a switch
            if switch_requested and time_since_switch >= self.min_green_time:
                # Initiate phase switch with yellow light
                self.current_phase = 1 - self.current_phase  # Toggle between 0 and 1
                self.last_switch_time = self.current_time
                self.yellow_timer = self.yellow_time
                phase_is_active = False  # Yellow light means all red
            else:
                phase_is_active = True

        # Phase 0 (NS) Green: Lanes 0 and 1 flow
        # Phase 1 (EW) Green: Lanes 2 and 3 flow
        ns_green = (self.current_phase == 0) and phase_is_active
        ew_green = (self.current_phase == 1) and phase_is_active

        lane_queues = []
        total_step_wait = 0
        total_throughput = 0

        # Update all lanes
        # Lanes 0,1 are NS. Lanes 2,3 are EW.
        for i, lane in enumerate(self.lanes):
            is_green = False
            if i in [0, 1] and ns_green: is_green = True
            if i in [2, 3] and ew_green: is_green = True
            
            q_len, discharged_vehicle = lane.step(is_green, self.current_time)
            lane_queues.append(q_len)
            total_step_wait += q_len
            if discharged_vehicle:
                total_throughput += 1

        # Reward: Negative total waiting time (Minimize delay)
        # Apply penalty for switching
        reward = -1.0 * total_step_wait
        if self.yellow_timer == self.yellow_time:  # Just switched
            reward -= self.switching_penalty

        # Check termination
        terminated = self.current_time >= self.max_steps
        truncated = False

        info = {
            "throughput": total_throughput,
            "total_wait": total_step_wait,
            "yellow_timer": self.yellow_timer
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
