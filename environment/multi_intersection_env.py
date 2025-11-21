import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.lane import Lane


class MultiIntersectionEnv(gym.Env):
    """
    A two-intersection environment where:
    - Intersection 1: North-South (N, S, E, W lanes)
    - Intersection 2: North-South (N, S, E, W lanes)
    - Link: Intersection 1 East lane feeds into Intersection 2 West lane
    
    Observation: [Q_N1, Q_S1, Q_E1, Q_W1, Phase1, Q_N2, Q_S2, Q_E2, Q_W2, Phase2]
    Action: [action_for_intersection_1, action_for_intersection_2]
    """
    
    def __init__(self, min_green_time=5, yellow_time=2, switching_penalty=10.0):
        super(MultiIntersectionEnv, self).__init__()
        
        # Action: MultiDiscrete([2, 2])
        # Each intersection: 0 = Keep phase, 1 = Switch phase
        self.action_space = spaces.MultiDiscrete([2, 2])
        
        # Observation: 10 values (5 per intersection)
        self.max_cars = 50
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_cars,
            shape=(10,),
            dtype=np.float32
        )
        
        # Create two intersections with 4 lanes each
        # Intersection 1: lanes 0-3 (N, S, E, W)
        # Intersection 2: lanes 4-7 (N, S, E, W)
        self.lanes = [Lane(i) for i in range(8)]
        
        # Phase tracking for each intersection
        self.current_phase = [0, 0]  # Start with NS Green for both
        
        # Switching constraints for each intersection
        self.min_green_time = min_green_time
        self.yellow_time = yellow_time
        self.switching_penalty = switching_penalty
        self.last_switch_time = [-self.min_green_time, -self.min_green_time]
        self.yellow_timer = [0, 0]
        
        self.current_time = 0
        self.max_steps = 1000
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.current_phase = [0, 0]
        self.last_switch_time = [-self.min_green_time, -self.min_green_time]
        self.yellow_timer = [0, 0]
        for lane in self.lanes:
            lane.reset()
        
        return self._get_obs(), {}
    
    def _link_intersections(self):
        """
        Route discharged vehicles from Intersection 1 East (lane 2)
        to Intersection 2 West (lane 7).
        This is called after both intersections have been stepped.
        """
        # In a more complex model, we'd track vehicles and apply routing logic here.
        # For now, this is a placeholder for future enhancement.
        pass
    
    def step(self, action):
        self.current_time += 1
        
        total_reward = 0
        total_throughput = 0
        total_wait = 0
        
        # Process each intersection separately
        for intersection_idx in range(2):
            action_val = action[intersection_idx]
            lane_offset = intersection_idx * 4
            
            # Handle yellow light timer
            if self.yellow_timer[intersection_idx] > 0:
                self.yellow_timer[intersection_idx] -= 1
                phase_is_active = False
            else:
                # Action interpretation: 0 = Keep, 1 = Switch
                switch_requested = (action_val == 1)
                time_since_switch = self.current_time - self.last_switch_time[intersection_idx]
                
                if switch_requested and time_since_switch >= self.min_green_time:
                    self.current_phase[intersection_idx] = 1 - self.current_phase[intersection_idx]
                    self.last_switch_time[intersection_idx] = self.current_time
                    self.yellow_timer[intersection_idx] = self.yellow_time
                    phase_is_active = False
                else:
                    phase_is_active = True
            
            # Determine which lanes are green
            ns_green = (self.current_phase[intersection_idx] == 0) and phase_is_active
            ew_green = (self.current_phase[intersection_idx] == 1) and phase_is_active
            
            # Update lanes for this intersection
            step_throughput = 0
            step_wait = 0
            
            for lane_idx in range(4):
                global_lane_idx = lane_offset + lane_idx
                
                is_green = False
                if lane_idx in [0, 1] and ns_green:  # N and S lanes
                    is_green = True
                if lane_idx in [2, 3] and ew_green:  # E and W lanes
                    is_green = True
                
                q_len, discharged = self.lanes[global_lane_idx].step(
                    is_green, self.current_time
                )
                step_throughput += discharged
                step_wait += q_len
            
            # Accumulate metrics
            intersection_reward = -1.0 * step_wait
            if self.yellow_timer[intersection_idx] == self.yellow_time:
                intersection_reward -= self.switching_penalty
            
            total_reward += intersection_reward
            total_throughput += step_throughput
            total_wait += step_wait
        
        # Link intersections (transfer vehicles from Int1 E to Int2 W)
        self._link_intersections()
        
        terminated = self.current_time >= self.max_steps
        truncated = False
        
        info = {
            "throughput": total_throughput,
            "total_wait": total_wait,
            "intersection_1_phase": self.current_phase[0],
            "intersection_2_phase": self.current_phase[1],
        }
        
        return self._get_obs(), total_reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Observation: [Q_N1, Q_S1, Q_E1, Q_W1, Phase1, Q_N2, Q_S2, Q_E2, Q_W2, Phase2]
        """
        obs = []
        for i in range(8):
            obs.append(len(self.lanes[i].queue))
        obs.append(self.current_phase[0])
        obs.append(self.current_phase[1])
        
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        obs = self._get_obs()
        print(f"Time: {self.current_time}")
        print(f"Intersection 1 - Phase: {'NS' if obs[4]==0 else 'EW'} | Queues: N={obs[0]}, S={obs[1]}, E={obs[2]}, W={obs[3]}")
        print(f"Intersection 2 - Phase: {'NS' if obs[9]==0 else 'EW'} | Queues: N={obs[5]}, S={obs[6]}, E={obs[7]}, W={obs[8]}")
        print()
