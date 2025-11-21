from collections import deque
import numpy as np
from environment.vehicle import Vehicle

class Lane:
    def __init__(self, id, arrival_rate=0.3):
        self.id = id
        self.queue = deque()
        self.arrival_rate = arrival_rate  # Prob of new car per step
        self.total_wait_time = 0

    def step(self, is_green, current_time):
        # 1. Spawn new vehicles (stochastic)
        if np.random.rand() < self.arrival_rate:
            self.queue.append(Vehicle(current_time))

        # 2. Discharge vehicles if green
        # Assuming 1 car passes per step if green (simplified flow)
        discharged_count = 0
        if is_green and len(self.queue) > 0:
            departed_car = self.queue.popleft()
            wait_time = current_time - departed_car.arrival_time
            discharged_count = 1
            # In a real sim, you might track this for metrics
        
        # 3. Calculate wait time for this step
        # Simple metric: number of cars currently in queue = instantaneous delay
        current_queue_len = len(self.queue)
        self.total_wait_time += current_queue_len
        
        # Return tuple: (queue_length, discharged_count)
        return current_queue_len, discharged_count

    def reset(self):
        self.queue.clear()
        self.total_wait_time = 0
