"""
Regular Traffic Light Controller (Baseline)

Implements a classical traffic light control system with:
- Fixed cycle time
- Phase sequencing (NS -> EW -> NS...)
- Offset coordination between two intersections
- Conflict monitoring

This controller is designed to be fast-running yet realistic, serving as a
baseline benchmark for RL experiments.
"""

import numpy as np
from typing import Tuple


class RegularTrafficLightController:
    """
    A regular (classical) traffic light controller for two connected intersections.
    
    The controller uses:
    - Fixed cycle time: Total duration for one complete phase sequence
    - Phase sequencing: Alternates between NS and EW green phases
    - Offset coordination: Time difference between when intersections start cycles
    - Single-alternating plan: Creates a "green wave" effect
    
    Parameters:
    -----------
    cycle_time : int
        Total duration of one complete cycle (NS green + yellow + EW green + yellow)
        Default: 30 steps
    green_time_ns : int
        Duration of North-South green phase (default: 12 steps)
    green_time_ew : int
        Duration of East-West green phase (default: 12 steps)
    yellow_time : int
        Duration of yellow/all-red phase (default: 2 steps)
    offset : int
        Time offset between intersection 1 and intersection 2 cycle starts
        If None, uses single-alternating plan (offset = cycle_time / 2)
        Default: None (auto-calculated)
    travel_time : int
        Estimated travel time between intersections (used to calculate optimal offset)
        Default: 8 steps
    """
    
    def __init__(
        self,
        cycle_time: int = 30,
        green_time_ns: int = 12,
        green_time_ew: int = 12,
        yellow_time: int = 2,
        offset: int = None,
        travel_time: int = 8
    ):
        self.cycle_time = cycle_time
        self.green_time_ns = green_time_ns
        self.green_time_ew = green_time_ew
        self.yellow_time = yellow_time
        
        # Validate cycle time matches phase durations
        calculated_cycle = green_time_ns + yellow_time + green_time_ew + yellow_time
        if cycle_time != calculated_cycle:
            print(f"Warning: cycle_time ({cycle_time}) doesn't match "
                  f"calculated cycle ({calculated_cycle}). Using calculated cycle.")
            self.cycle_time = calculated_cycle
        
        # Calculate offset if not provided (use corrected cycle_time)
        if offset is None:
            # Single-alternating plan: offset by half cycle time for green wave
            # Or use travel_time - 1 for platoon coordination
            self.offset = min(self.cycle_time // 2, max(1, travel_time - 1))
        else:
            self.offset = offset
        
        # Phase state tracking
        self.cycle_start_time_1 = 0  # When intersection 1 started its current cycle
        self.cycle_start_time_2 = -self.offset  # Intersection 2 starts offset steps later
        
        # Current phase within cycle for each intersection
        # 0 = NS green, 1 = yellow after NS, 2 = EW green, 3 = yellow after EW
        self.phase_state_1 = 0
        self.phase_state_2 = 0
        
        # Time within current phase
        self.phase_timer_1 = 0
        self.phase_timer_2 = 0
    
    def reset(self):
        """Reset the controller to initial state."""
        self.cycle_start_time_1 = 0
        self.cycle_start_time_2 = -self.offset
        self.phase_state_1 = 0
        self.phase_state_2 = 0
        self.phase_timer_1 = 0
        self.phase_timer_2 = 0
    
    def get_action(self, current_time: int) -> Tuple[int, int]:
        """
        Get the action for both intersections at the current time step.
        
        Parameters:
        -----------
        current_time : int
            Current simulation time step
        
        Returns:
        --------
        action : Tuple[int, int]
            Action for (intersection_1, intersection_2)
            Each action: 0 = Keep phase, 1 = Switch phase
        """
        actions = [0, 0]  # Default: keep current phase
        
        # Update intersection 1
        actions[0] = self._get_intersection_action(
            current_time, 
            self.cycle_start_time_1,
            self.phase_state_1,
            self.phase_timer_1
        )
        
        # Update intersection 2
        actions[1] = self._get_intersection_action(
            current_time,
            self.cycle_start_time_2,
            self.phase_state_2,
            self.phase_timer_2
        )
        
        # Update internal state
        self._update_state(current_time, actions)
        
        return tuple(actions)
    
    def _get_intersection_action(
        self, 
        current_time: int,
        cycle_start: int,
        phase_state: int,
        phase_timer: int
    ) -> int:
        """
        Determine action for a single intersection based on timing.
        
        Returns:
        --------
        action : int
            0 = Keep phase, 1 = Switch phase
        """
        # Calculate time within current cycle
        time_in_cycle = (current_time - cycle_start) % self.cycle_time
        
        # Determine expected phase based on cycle timing
        if time_in_cycle < self.green_time_ns:
            expected_phase_state = 0  # NS green
            expected_timer = time_in_cycle
        elif time_in_cycle < self.green_time_ns + self.yellow_time:
            expected_phase_state = 1  # Yellow after NS
            expected_timer = time_in_cycle - self.green_time_ns
        elif time_in_cycle < self.green_time_ns + self.yellow_time + self.green_time_ew:
            expected_phase_state = 2  # EW green
            expected_timer = time_in_cycle - (self.green_time_ns + self.yellow_time)
        else:
            expected_phase_state = 3  # Yellow after EW
            expected_timer = time_in_cycle - (self.green_time_ns + self.yellow_time + self.green_time_ew)
        
        # Check if we need to switch
        # Switch when transitioning between phases
        if phase_state != expected_phase_state:
            return 1  # Switch phase
        
        return 0  # Keep phase
    
    def _update_state(self, current_time: int, actions: Tuple[int, int]):
        """Update internal state after determining actions."""
        # Update intersection 1
        if actions[0] == 1:
            # Phase switch occurred
            self.phase_state_1 = (self.phase_state_1 + 1) % 4
            self.phase_timer_1 = 0
            
            # If we completed a full cycle, reset cycle start
            if self.phase_state_1 == 0:
                self.cycle_start_time_1 = current_time
        else:
            self.phase_timer_1 += 1
        
        # Update intersection 2
        if actions[1] == 1:
            # Phase switch occurred
            self.phase_state_2 = (self.phase_state_2 + 1) % 4
            self.phase_timer_2 = 0
            
            # If we completed a full cycle, reset cycle start
            if self.phase_state_2 == 0:
                self.cycle_start_time_2 = current_time
        else:
            self.phase_timer_2 += 1
    
    def get_current_phases(self) -> Tuple[int, int]:
        """
        Get the current phase for each intersection.
        
        Returns:
        --------
        phases : Tuple[int, int]
            (phase_1, phase_2) where 0 = NS green, 1 = EW green
            Note: During yellow, returns the phase that was just active
        """
        # Map phase state to actual phase (0 = NS, 1 = EW)
        phase_1 = 0 if self.phase_state_1 in [0, 1] else 1
        phase_2 = 0 if self.phase_state_2 in [0, 1] else 1
        
        return (phase_1, phase_2)
    
    def get_info(self) -> dict:
        """
        Get controller information for debugging/monitoring.
        
        Returns:
        --------
        info : dict
            Dictionary with controller state information
        """
        return {
            "cycle_time": self.cycle_time,
            "offset": self.offset,
            "phase_state_1": self.phase_state_1,
            "phase_state_2": self.phase_state_2,
            "phase_timer_1": self.phase_timer_1,
            "phase_timer_2": self.phase_timer_2,
            "cycle_start_1": self.cycle_start_time_1,
            "cycle_start_2": self.cycle_start_time_2,
        }


class FixedTimeController:
    """
    Simplified fixed-time controller that doesn't track internal state.
    This version is faster and can be used when you just need actions.
    
    Parameters are the same as RegularTrafficLightController.
    """
    
    def __init__(
        self,
        cycle_time: int = 30,
        green_time_ns: int = 12,
        green_time_ew: int = 12,
        yellow_time: int = 2,
        offset: int = None,
        travel_time: int = 8
    ):
        self.cycle_time = cycle_time
        self.green_time_ns = green_time_ns
        self.green_time_ew = green_time_ew
        self.yellow_time = yellow_time
        
        # Validate cycle time - it should match the sum of all phase durations
        calculated_cycle = green_time_ns + yellow_time + green_time_ew + yellow_time
        if cycle_time != calculated_cycle:
            print(f"Warning: cycle_time ({cycle_time}) doesn't match calculated cycle ({calculated_cycle}). "
                  f"Using calculated cycle.")
            self.cycle_time = calculated_cycle
        else:
            self.cycle_time = cycle_time
        
        # Calculate offset (use corrected cycle_time)
        if offset is None:
            self.offset = min(self.cycle_time // 2, max(1, travel_time - 1))
        else:
            self.offset = offset
    
    def reset(self):
        """Reset (no-op for stateless controller)."""
        pass
    
    def get_action(self, current_time: int) -> Tuple[int, int]:
        """
        Get action for both intersections based solely on current time.
        This is faster as it doesn't maintain internal state.
        
        Parameters:
        -----------
        current_time : int
            Current simulation time step
        
        Returns:
        --------
        action : Tuple[int, int]
            Action for (intersection_1, intersection_2)
        """
        # Calculate time within cycle for each intersection
        # Int2's cycle starts 'offset' steps after Int1's cycle
        time_in_cycle_1 = current_time % self.cycle_time
        time_in_cycle_2 = (current_time - self.offset) % self.cycle_time
        
        # Determine if we're at a phase transition point
        action_1 = self._should_switch(time_in_cycle_1)
        action_2 = self._should_switch(time_in_cycle_2)
        
        return (action_1, action_2)
    
    def _should_switch(self, time_in_cycle: int) -> int:
        """
        Determine if a switch should occur at this time in the cycle.
        
        Returns:
        --------
        action : int
            0 = Keep, 1 = Switch
        """
        # Phase boundaries where switches occur:
        # - End of NS green (start yellow) at green_time_ns
        # - End of yellow after NS (start EW green) at green_time_ns + yellow_time
        # - End of EW green (start yellow) at green_time_ns + yellow_time + green_time_ew
        # - End of yellow after EW (start NS green) at cycle_time (wraps to 0)
        
        # Check if we're exactly at a switch point
        # Note: time_in_cycle is already modulo cycle_time, so 0 means start of cycle
        last_switch_point = self.green_time_ns + self.yellow_time + self.green_time_ew + self.yellow_time
        
        if time_in_cycle == 0:
            return 1  # Start of cycle (NS green begins) - also handles wrap-around from last switch
        elif time_in_cycle == self.green_time_ns:
            return 1  # NS green ends, yellow begins
        elif time_in_cycle == self.green_time_ns + self.yellow_time:
            return 1  # Yellow ends, EW green begins
        elif time_in_cycle == self.green_time_ns + self.yellow_time + self.green_time_ew:
            return 1  # EW green ends, yellow begins
        elif time_in_cycle == last_switch_point % self.cycle_time:
            return 1  # Yellow ends, NS green begins (completes cycle)
        
        return 0
    
    def get_current_phases(self, current_time: int) -> Tuple[int, int]:
        """
        Get current phase for each intersection.
        
        Returns:
        --------
        phases : Tuple[int, int]
            (phase_1, phase_2) where 0 = NS green, 1 = EW green
        """
        time_in_cycle_1 = current_time % self.cycle_time
        time_in_cycle_2 = (current_time + self.offset) % self.cycle_time
        
        phase_1 = self._get_phase_from_time(time_in_cycle_1)
        phase_2 = self._get_phase_from_time(time_in_cycle_2)
        
        return (phase_1, phase_2)
    
    def _get_phase_from_time(self, time_in_cycle: int) -> int:
        """Get phase (0=NS, 1=EW) from time within cycle."""
        ns_end = self.green_time_ns + self.yellow_time
        
        if time_in_cycle < ns_end:
            return 0  # NS phase (including yellow)
        else:
            return 1  # EW phase (including yellow)
    
    def get_info(self) -> dict:
        """Get controller information."""
        return {
            "cycle_time": self.cycle_time,
            "offset": self.offset,
            "green_time_ns": self.green_time_ns,
            "green_time_ew": self.green_time_ew,
            "yellow_time": self.yellow_time,
        }

