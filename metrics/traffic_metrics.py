import time
import json
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List

# Fixes int64 json encoding
# https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class TrafficMetrics:
    """
    Tracks:
    - Reward and performance metrics
    - Queue-based metrics (average/max queue lengths, growth rates, variance)
    - Wait time metrics (per-vehicle, average, cumulative)
    - Throughput metrics (vehicles served, discharge rate, efficiency)
    - Phase metrics (durations, switches, green time utilization)
    - Efficiency metrics (system efficiency, fairness index, congestion level)
    - Stability metrics (queue oscillations, phase switches)
    - Performance over time (time series analysis)
    """

    def __init__(self, num_lanes=4, window_size=100, env_type="single"):
        self.num_lanes = num_lanes
        self.window_size = window_size
        self.env_type = env_type

        # Basic counters
        self.total_reward = 0.0
        self.step_count = 0
        self.episode_count = 0

        # Queue metrics
        self.queue_lengths = defaultdict(list)  # Per-lane queue history
        self.max_queue_lengths = defaultdict(list)  # Max queue per episode per lane
        self.queue_growth_rates = defaultdict(list)  # Rate of queue growth per lane
        self.queue_history = []  # Full queue state history
        self.queue_overflow_count = [0] * num_lanes
        self.queue_threshold = 40

        # Wait time metrics
        self.vehicle_wait_times = []  # Individual vehicle wait times
        self.cumulative_wait_time = 0.0
        self.average_wait_times = []  # Average wait time per episode
        self.max_wait_time = 0.0
        self.wait_time_per_step = []

        # Throughput metrics
        self.vehicles_served = 0
        self.throughput_per_step = []
        self.throughput_efficiency = []  # vehicles served per green second
        self.total_vehicles_arrived = 0
        self.throughput_rate_history = []

        # Phase metrics
        self.phase_durations = defaultdict(list)
        self.phase_changes = 0
        self.phase_switch_frequency = []  # Switches per episode
        self.green_time_utilization = []  # Efficiency of green time usage
        self.current_phase = None
        self.phase_start_step = 0
        self.phase_history = []
        self.yellow_time_count = 0

        # Intersection-specific metrics (for multi-intersection)
        if self.env_type == "multi":
            self.intersection_metrics = {
                1: defaultdict(list),
                2: defaultdict(list)
            }
            self.coordination_efficiency = []
            self.phase_sync_count = 0
            self.phase_offset_history = []

        # Efficiency metrics
        self.system_efficiency = []  # Overall system efficiency per episode
        self.fairness_index = []  # Jain's fairness index across lanes
        self.congestion_level = []  # Congestion level per episode
        self.service_level = []  # Percentage of time with acceptable queues

        # Stability metrics
        self.queue_oscillation = []  # Queue instability measure
        self.queue_variance_history = []

        # Per-step raw data
        self.step_data = defaultdict(list)

        # Per-episode aggregated data
        self.episode_data = defaultdict(list)

        # Rolling windows for real-time monitoring
        self.rolling_queues = [deque(maxlen=window_size) for _ in range(num_lanes)]
        self.rolling_throughput = deque(maxlen=window_size)
        self.rolling_rewards = deque(maxlen=window_size)

        # Time tracking
        self.start_time = time.time()
        self.episode_start_time = time.time()
        self.episode_start_step = 0
        self.last_queue_state = None

        # Green time tracking
        self.green_time_by_phase = defaultdict(int)
        self.vehicles_served_by_phase = defaultdict(int)

    def update_step(self,
                    queues: List[int],
                    throughput: int,
                    wait_time: float,
                    phase: int,
                    reward: float,
                    action: int = None,
                    yellow_active: bool = False,
                    info: Dict = None):
        """
        Update metrics after each environment step.

        Args:
            queues: List of queue lengths for each lane
            throughput: Number of vehicles served this step
            wait_time: Total waiting time this step
            phase: Current traffic light phase
            reward: Reward received this step
            action: Action taken (optional)
            yellow_active: Whether yellow light is active
            info: Additional info dict from environment
        """
        self.step_count += 1

        # Basic tracking
        self.total_reward += reward
        self.cumulative_wait_time += wait_time
        self.vehicles_served += throughput

        # Store raw step data
        self.step_data['queues'].append(queues.copy())
        self.step_data['throughput'].append(throughput)
        self.step_data['wait_time'].append(wait_time)
        self.step_data['phase'].append(phase)
        self.step_data['reward'].append(reward)
        self.step_data['yellow_active'].append(yellow_active)

        if action is not None:
            self.step_data['action'].append(action)

        self.wait_time_per_step.append(wait_time)
        self.throughput_per_step.append(throughput)
        self.queue_history.append(queues.copy())
        self.phase_history.append(phase)

        # Update rolling windows
        self.rolling_rewards.append(reward)
        self.rolling_throughput.append(throughput)

        # Queue metrics
        for i, q in enumerate(queues):
            self.queue_lengths[i].append(q)
            self.rolling_queues[i].append(q)

            # Track overflows
            if q > self.queue_threshold:
                self.queue_overflow_count[i] += 1

        # Calculate queue growth rate
        if self.last_queue_state is not None:
            for i in range(self.num_lanes):
                growth = queues[i] - self.last_queue_state[i]
                self.queue_growth_rates[i].append(growth)

        self.last_queue_state = queues.copy()

        # Phase tracking
        if yellow_active:
            self.yellow_time_count += 1

        if self.current_phase is not None and self.current_phase != phase:
            # Phase changed
            duration = self.step_count - self.phase_start_step
            self.phase_durations[self.current_phase].append(duration)
            self.phase_changes += 1
            self.phase_start_step = self.step_count

        elif self.current_phase is None:
            # First step
            self.phase_start_step = self.step_count

        self.current_phase = phase

        # Track green time utilization
        if not yellow_active:
            self.green_time_by_phase[phase] += 1
            self.vehicles_served_by_phase[phase] += throughput

        # Calculate per-step efficiency
        if not yellow_active and self.green_time_by_phase[phase] > 0:
            efficiency = self.vehicles_served_by_phase[phase] / self.green_time_by_phase[phase]
            self.throughput_efficiency.append(efficiency)

    def update_vehicle_wait_time(self, wait_time: float):
        """Update individual vehicle wait time"""
        self.vehicle_wait_times.append(wait_time)
        self.max_wait_time = max(self.max_wait_time, wait_time)

    def finalize_episode(self) -> Dict:
        """
        Calculate episode-level metrics and reset step counters.

        Returns:
            Dictionary containing all episode metrics
        """
        self.episode_count += 1
        episode_length = self.step_count - self.episode_start_step
        episode_duration = time.time() - self.episode_start_time

        if episode_length == 0:
            return {}

        # Basic statistics
        metrics = {
            'episode_number': self.episode_count,
            'episode_length': episode_length,
            'episode_duration_sec': episode_duration,
            'steps_per_sec': episode_length / episode_duration if episode_duration > 0 else 0,
            'total_reward': self.total_reward,
            'mean_reward': self.total_reward / episode_length,
            'mean_reward_per_sec': self.total_reward / episode_duration if episode_duration > 0 else 0,
        }

        # Queue metrics
        # Calculate max queue length per lane during episode
        for lane in range(self.num_lanes):
            if lane in self.queue_lengths and self.queue_lengths[lane]:
                max_q = max(self.queue_lengths[lane])
                self.max_queue_lengths[lane].append(max_q)
                metrics[f'lane_{lane}_max_queue'] = max_q
                metrics[f'lane_{lane}_mean_queue'] = np.mean(self.queue_lengths[lane])
                metrics[f'lane_{lane}_std_queue'] = np.std(self.queue_lengths[lane])

                # Queue growth rate
                if lane in self.queue_growth_rates and self.queue_growth_rates[lane]:
                    metrics[f'lane_{lane}_mean_growth_rate'] = np.mean(self.queue_growth_rates[lane])
                    metrics[f'lane_{lane}_max_growth'] = max(self.queue_growth_rates[lane])

        # Overall queue metrics
        all_queues = np.array(self.step_data['queues'])
        metrics['mean_queue_length'] = np.mean(all_queues)
        metrics['max_queue_length'] = np.max(all_queues)
        metrics['queue_length_std'] = np.std(all_queues)
        metrics['total_queue_time'] = np.sum(all_queues)

        # Per-lane statistics
        lane_means = [np.mean(self.queue_lengths[i]) if self.queue_lengths[i] else 0
                      for i in range(self.num_lanes)]
        lane_maxs = [max(self.queue_lengths[i]) if self.queue_lengths[i] else 0
                     for i in range(self.num_lanes)]

        metrics['lane_queue_means'] = lane_means
        metrics['lane_queue_maxs'] = lane_maxs

        # Wait time metrics
        metrics['total_wait_time'] = self.cumulative_wait_time
        metrics['mean_wait_time'] = np.mean(self.wait_time_per_step) if self.wait_time_per_step else 0
        metrics['max_wait_time_step'] = max(self.wait_time_per_step) if self.wait_time_per_step else 0

        if self.vehicle_wait_times:
            avg_vehicle_wait = np.mean(self.vehicle_wait_times)
            self.average_wait_times.append(avg_vehicle_wait)
            metrics['avg_vehicle_wait_time'] = avg_vehicle_wait
            metrics['max_vehicle_wait_time'] = self.max_wait_time

        # Throughput metrics
        metrics['total_throughput'] = self.vehicles_served
        metrics['mean_throughput'] = np.mean(self.throughput_per_step) if self.throughput_per_step else 0
        metrics['throughput_rate'] = self.vehicles_served / episode_length

        if self.throughput_efficiency:
            metrics['mean_throughput_efficiency'] = np.mean(self.throughput_efficiency)

        # Phase metrics
        metrics['phase_switches'] = self.phase_changes
        self.phase_switch_frequency.append(self.phase_changes)

        # Phase duration statistics
        for phase, durations in self.phase_durations.items():
            if durations:
                metrics[f'phase_{phase}_mean_duration'] = np.mean(durations)
                metrics[f'phase_{phase}_total_time'] = sum(durations)
                metrics[f'phase_{phase}_count'] = len(durations)

        # Green time utilization
        total_green_time = sum(self.green_time_by_phase.values())
        if total_green_time > 0:
            green_util = (episode_length - self.yellow_time_count) / episode_length
            self.green_time_utilization.append(green_util)
            metrics['green_time_utilization'] = green_util

            # Per-phase efficiency
            for phase in self.green_time_by_phase:
                if self.green_time_by_phase[phase] > 0:
                    phase_efficiency = self.vehicles_served_by_phase[phase] / self.green_time_by_phase[phase]
                    metrics[f'phase_{phase}_efficiency'] = phase_efficiency

        # Efficiency metrics
        # Average delay per vehicle
        if self.vehicles_served > 0:
            metrics['avg_delay_per_vehicle'] = self.cumulative_wait_time / self.vehicles_served
        else:
            metrics['avg_delay_per_vehicle'] = 0

        # System efficiency
        total_queue_time = np.sum(all_queues)
        if total_queue_time > 0:
            sys_eff = self.vehicles_served / total_queue_time
            self.system_efficiency.append(sys_eff)
            metrics['system_efficiency'] = sys_eff

        # Jain's fairness index across lanes
        fairness = self._calculate_jains_fairness_index(lane_means)
        self.fairness_index.append(fairness)
        metrics['fairness_index'] = fairness

        # Congestion level
        congestion = np.mean(all_queues) / self.queue_threshold
        self.congestion_level.append(congestion)
        metrics['congestion_level'] = congestion

        # Service level (percentage of time with acceptable queue lengths)
        acceptable_steps = sum(1 for queues in self.step_data['queues']
                               if all(q <= self.queue_threshold for q in queues))
        service_lvl = acceptable_steps / episode_length
        self.service_level.append(service_lvl)
        metrics['service_level'] = service_lvl

        # Stability metrics
        # Queue imbalance
        metrics['queue_imbalance'] = self._calculate_queue_imbalance()

        # Queue oscillation
        oscillation = self._calculate_queue_oscillation()
        self.queue_oscillation.append(oscillation)
        metrics['queue_oscillation'] = oscillation

        # Queue overflow rate
        metrics['queue_overflow_rate'] = sum(self.queue_overflow_count) / (episode_length * self.num_lanes)

        # Queue variance over time
        queue_variance = np.var(all_queues)
        self.queue_variance_history.append(queue_variance)
        metrics['queue_variance'] = queue_variance

        # Multi-intersection specific metrics
        if self.env_type == "multi":
            metrics.update(self._calculate_coordination_metrics(episode_length))

        # Store episode metrics
        for key, value in metrics.items():
            self.episode_data[key].append(value)

        # Reset for next episode
        self._reset_episode()

        return metrics

    def _calculate_jains_fairness_index(self, values: List[float]) -> float:
        """
        Calculate Jain's Fairness Index.
        Returns value between 0 and 1, where 1 is perfectly fair.
        """
        if not values or len(values) == 0:
            return 1.0

        values = np.array(values)
        n = len(values)

        if np.sum(values) == 0:
            return 1.0

        numerator = np.sum(values) ** 2
        denominator = n * np.sum(values ** 2)

        if denominator == 0:
            return 1.0

        return numerator / denominator

    def _calculate_queue_imbalance(self) -> float:
        """
        Calculate queue imbalance across lanes using coefficient of variation.
        Lower values indicate more balanced queue distribution.
        """
        if not self.step_data['queues']:
            return 0.0

        imbalances = []
        for queues in self.step_data['queues']:
            mean_q = np.mean(queues)
            if mean_q > 0:
                cv = np.std(queues) / mean_q
                imbalances.append(cv)

        return np.mean(imbalances) if imbalances else 0.0

    def _calculate_queue_oscillation(self) -> float:
        """
        Calculate queue oscillation (instability measure).
        Measures how much queues fluctuate over time.
        """
        if len(self.step_data['queues']) < 2:
            return 0.0

        oscillations = []
        for lane_idx in range(self.num_lanes):
            if lane_idx in self.queue_lengths and len(self.queue_lengths[lane_idx]) > 1:
                lane_queues = self.queue_lengths[lane_idx]
                # Calculate sum of absolute differences
                diffs = [abs(lane_queues[i] - lane_queues[i-1])
                         for i in range(1, len(lane_queues))]
                oscillations.append(np.mean(diffs) if diffs else 0.0)

        return np.mean(oscillations) if oscillations else 0.0

    def _calculate_coordination_metrics(self, episode_length: int) -> Dict:
        metrics = {}

        if episode_length > 0:
            # Phase synchronization rate
            sync_rate = self.phase_sync_count / episode_length
            self.coordination_efficiency.append(sync_rate)
            metrics['phase_sync_rate'] = sync_rate

            # Mean phase offset
            if self.phase_offset_history:
                metrics['mean_phase_offset'] = np.mean(self.phase_offset_history)
                metrics['phase_offset_std'] = np.std(self.phase_offset_history)

        return metrics

    def update_multi_intersection_step(self, obs, actions, reward, info):
        """
        Update metrics for multi-intersection system.

        Args:
            obs: Observation array
            actions: Action array [action_int1, action_int2]
            reward: Total reward
            info: Info dictionary from environment
        """
        # Determine lanes per intersection
        lanes_per_int = self.num_lanes // 2  # Assuming 2 intersections
        num_intersections = 2

        # Extract overall queues
        all_queues = obs[:self.num_lanes].tolist()

        # Extract phases
        phase_start_idx = self.num_lanes
        phases = [int(obs[phase_start_idx + i]) for i in range(num_intersections)]

        # Track coordination
        if len(set(phases)) == 1:
            self.phase_sync_count += 1

        if len(phases) == 2:
            self.phase_offset_history.append(abs(phases[0] - phases[1]))

        # Update per-intersection metrics
        for i in range(num_intersections):
            start_idx = i * lanes_per_int
            end_idx = start_idx + lanes_per_int
            int_queues = all_queues[start_idx:end_idx]

            # Store intersection-specific data
            self.intersection_metrics[i+1]['queues'].append(int_queues)
            self.intersection_metrics[i+1]['phase'].append(phases[i])
            self.intersection_metrics[i+1]['action'].append(actions[i] if hasattr(actions, '__len__') else None)

        # Update overall metrics
        throughput = info.get('throughput', 0)
        wait_time = info.get('total_wait', 0)

        self.update_step(
            queues=all_queues,
            throughput=throughput,
            wait_time=wait_time,
            phase=phases[0],  # Use first intersection's phase for overall tracking
            reward=reward,
            action=actions[0] if hasattr(actions, '__len__') else actions,
            info=info
        )

    def _reset_episode(self):
        """Reset episode-level counters"""
        self.total_reward = 0.0
        self.cumulative_wait_time = 0.0
        self.vehicles_served = 0
        self.phase_changes = 0
        self.yellow_time_count = 0

        # Reset tracking structures
        self.step_data = defaultdict(list)
        self.queue_lengths = defaultdict(list)
        self.queue_growth_rates = defaultdict(list)
        self.queue_overflow_count = [0] * self.num_lanes
        self.vehicle_wait_times = []
        self.wait_time_per_step = []
        self.throughput_per_step = []
        self.throughput_efficiency = []
        self.phase_durations = defaultdict(list)
        self.green_time_by_phase = defaultdict(int)
        self.vehicles_served_by_phase = defaultdict(int)
        self.queue_history = []
        self.phase_history = []

        self.current_phase = None
        self.last_queue_state = None
        self.max_wait_time = 0.0

        self.episode_start_step = self.step_count
        self.episode_start_time = time.time()

        # Reset multi-intersection tracking
        if self.env_type == "multi":
            self.phase_sync_count = 0
            self.phase_offset_history = []
            for intersection in self.intersection_metrics.values():
                for key in intersection:
                    intersection[key].clear()

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics across all episodes.

        Returns:
            Dictionary with mean, std, min, max for key metrics
        """
        summary = {}

        key_metrics = [
            'total_reward', 'mean_queue_length', 'max_queue_length',
            'total_throughput', 'mean_wait_time', 'phase_switches',
            'queue_imbalance', 'queue_oscillation', 'service_level',
            'avg_delay_per_vehicle', 'throughput_rate', 'system_efficiency',
            'fairness_index', 'congestion_level', 'green_time_utilization'
        ]

        for metric in key_metrics:
            if metric in self.episode_data and self.episode_data[metric]:
                data = self.episode_data[metric]
                summary[metric] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'median': np.median(data)
                }

        # Add cumulative statistics
        summary['total_episodes'] = self.episode_count
        summary['total_steps'] = self.step_count
        summary['total_time_sec'] = time.time() - self.start_time
        summary['avg_steps_per_episode'] = self.step_count / self.episode_count if self.episode_count > 0 else 0

        return summary

    def get_time_series(self, metric: str, intersection: int = None) -> List:
        """
        Get time series data for a specific metric.

        Args:
            metric: Name of the metric
            intersection: If multi-intersection, specify which intersection (1 or 2)
        """
        if intersection is not None and self.env_type == "multi":
            return self.intersection_metrics[intersection].get(metric, [])
        return self.step_data.get(metric, [])

    def get_rolling_average(self, metric: str, window: int = None) -> np.ndarray:
        """Get rolling average for a metric"""
        if window is None:
            window = self.window_size

        data = self.episode_data.get(metric, [])
        if len(data) < window:
            return np.array(data)

        return np.convolve(data, np.ones(window)/window, mode='valid')

    def print_summary(self, detailed: bool = True):
        """Print a formatted summary of metrics."""
        summary = self.get_summary_statistics()

        print("\n" + "=" * 80)
        print("TRAFFIC METRICS SUMMARY")
        print("=" * 80)

        # Basic info
        print(f"\nSimulation Info:")
        print(f"  Total Episodes: {summary['total_episodes']}")
        print(f"  Total Steps: {summary['total_steps']}")
        print(f"  Total Time: {summary['total_time_sec']:.2f} seconds")
        print(f"  Avg Steps/Episode: {summary['avg_steps_per_episode']:.1f}")
        print(f"  Environment Type: {self.env_type}")

        # Performance metrics
        print(f"\n{'─'*80}")
        print("Performance Metrics:")
        print(f"{'─'*80}")

        perf_metrics = ['total_reward', 'mean_queue_length', 'max_queue_length',
                        'total_throughput', 'mean_wait_time']

        for metric in perf_metrics:
            if metric in summary:
                stats = summary[metric]
                metric_name = metric.replace('_', ' ').title()
                print(f"{metric_name:.<50} {stats['mean']:>12.2f} ± {stats['std']:>8.2f}")
                if detailed:
                    print(f"{'  (min/median/max)':<50} {stats['min']:>12.2f} / "
                          f"{stats['median']:>8.2f} / {stats['max']:>8.2f}")

        # Efficiency metrics
        print(f"\n{'─'*80}")
        print("Efficiency Metrics:")
        print(f"{'─'*80}")

        eff_metrics = ['system_efficiency', 'throughput_rate', 'avg_delay_per_vehicle',
                       'green_time_utilization', 'service_level']

        for metric in eff_metrics:
            if metric in summary:
                stats = summary[metric]
                metric_name = metric.replace('_', ' ').title()
                print(f"{metric_name:.<50} {stats['mean']:>12.4f} ± {stats['std']:>8.4f}")

        # Fairness and Stability
        print(f"\n{'─'*80}")
        print("Fairness & Stability Metrics:")
        print(f"{'─'*80}")

        fair_metrics = ['fairness_index', 'queue_imbalance', 'queue_oscillation',
                        'congestion_level', 'phase_switches']

        for metric in fair_metrics:
            if metric in summary:
                stats = summary[metric]
                metric_name = metric.replace('_', ' ').title()
                print(f"{metric_name:.<50} {stats['mean']:>12.4f} ± {stats['std']:>8.4f}")

        # Multi-intersection specific
        if self.env_type == "multi":
            print(f"\n{'─'*80}")
            print("Coordination Metrics:")
            print(f"{'─'*80}")

            coord_metrics = ['phase_sync_rate', 'mean_phase_offset']
            for metric in coord_metrics:
                if metric in self.episode_data and self.episode_data[metric]:
                    data = self.episode_data[metric]
                    metric_name = metric.replace('_', ' ').title()
                    print(f"{metric_name:.<50} {np.mean(data):>12.4f} ± {np.std(data):>8.4f}")

        # Lane-specific statistics (from last episode)
        if detailed and self.episode_data.get('lane_queue_means'):
            print(f"\n{'─'*80}")
            print("Lane-Specific Statistics (Last Episode):")
            print(f"{'─'*80}")

            last_means = self.episode_data['lane_queue_means'][-1] if self.episode_data['lane_queue_means'] else []
            last_maxs = self.episode_data['lane_queue_maxs'][-1] if self.episode_data['lane_queue_maxs'] else []

            lane_names = ['North', 'South', 'East', 'West']
            for i in range(min(len(lane_names), len(last_means))):
                print(f"  {lane_names[i]:.<45} Mean: {last_means[i]:>8.2f}, Max: {last_maxs[i]:>8.2f}")

        print("=" * 80 + "\n")

    def print_current_status(self):
        """Print current rolling averages for monitoring during training."""
        if len(self.rolling_rewards) == 0:
            return

        avg_reward = np.mean(self.rolling_rewards)
        avg_throughput = np.mean(self.rolling_throughput) if self.rolling_throughput else 0

        avg_queues = [np.mean(q) if len(q) > 0 else 0 for q in self.rolling_queues]

        print(f"[Step {self.step_count}] "
              f"Reward: {avg_reward:>8.2f} | "
              f"Throughput: {avg_throughput:>6.2f} | "
              f"Avg Queue: {np.mean(avg_queues):>6.2f}")

    def save_to_file(self, filepath: str):
        """Save metrics to a file."""
        # Prepare data for JSON serialization
        data = {
            'summary': self.get_summary_statistics(),
            'episode_data': {k: [float(v) if isinstance(v, (np.float32, np.float64)) else v
                                 for v in vals]
                             for k, vals in self.episode_data.items()},
            'config': {
                'num_lanes': self.num_lanes,
                'window_size': self.window_size,
                'env_type': self.env_type,
                'queue_threshold': self.queue_threshold
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NpEncoder)

        print(f"Metrics saved to {filepath}")

    def load_from_file(self, filepath: str):
        """Load metrics from a file."""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.episode_data = defaultdict(list, data['episode_data'])
        config = data.get('config', {})

        self.num_lanes = config.get('num_lanes', self.num_lanes)
        self.window_size = config.get('window_size', self.window_size)
        self.env_type = config.get('env_type', self.env_type)
        self.queue_threshold = config.get('queue_threshold', self.queue_threshold)

        print(f"Metrics loaded from {filepath}")


class MultiIntersectionMetrics(TrafficMetrics):
    """
    Extended metrics tracker specifically for multi-intersection environments.
    Inherits from TrafficMetrics and adds intersection-specific tracking(for now not so much)
    """

    def __init__(self, num_intersections=2, lanes_per_intersection=4, window_size=100):
        super().__init__(
            num_lanes=num_intersections * lanes_per_intersection,
            window_size=window_size,
            env_type="multi"
        )

        self.num_intersections = num_intersections
        self.lanes_per_intersection = lanes_per_intersection


# Convenience function to create appropriate metrics tracker
def create_metrics_tracker(env_type="single", **kwargs):
    """
    Factory function to create appropriate metrics tracker.

    Args:
        env_type: "single" or "multi"
        **kwargs: Additional arguments passed to metrics constructor

    Returns:
        TrafficMetrics or MultiIntersectionMetrics instance
    """
    if env_type == "multi":
        return MultiIntersectionMetrics(**kwargs)
    else:
        return TrafficMetrics(env_type=env_type, **kwargs)
