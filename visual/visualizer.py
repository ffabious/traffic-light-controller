import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np


class TrafficVisualizer:
    """
    Visualizes a single or multi-intersection traffic environment.
    """
    
    def __init__(self, env, mode='single', figsize=(12, 6)):
        """
        Args:
            env: The environment to visualize (TrafficIntersectionEnv or MultiIntersectionEnv)
            mode: 'single' or 'multi'
            figsize: Figure size for matplotlib
        """
        self.env = env
        self.mode = mode
        self.figsize = figsize
        self.fig = None
        self.axes = None
    
    def _draw_single_intersection(self, ax, queues, phase, title="Intersection"):
        """
        Draw a single intersection with queue visualization.
        
        Args:
            ax: Matplotlib axis
            queues: List of 4 queue lengths [N, S, E, W]
            phase: Current phase (0=NS Green, 1=EW Green)
        """
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Draw intersection roads
        # Vertical road (NS)
        ax.add_patch(Rectangle((-0.5, -2), 1, 4, fill=True, color='lightgray', zorder=1))
        # Horizontal road (EW)
        ax.add_patch(Rectangle((-2, -0.5), 4, 1, fill=True, color='lightgray', zorder=1))
        
        # Lane labels and queue visualization
        lanes = ['N', 'S', 'E', 'W']
        positions = [
            (0, 1.3, queues[0]),      # North
            (0, -1.3, queues[1]),     # South
            (1.3, 0, queues[2]),      # East
            (-1.3, 0, queues[3])      # West
        ]
        
        colors = ['green' if phase == 0 else 'red',  # N
                  'green' if phase == 0 else 'red',  # S
                  'green' if phase == 1 else 'red',  # E
                  'green' if phase == 1 else 'red']  # W
        
        # Draw traffic lights and queue bars
        light_positions = [
            (0.6, 0.6),   # NE corner
            (-0.6, -0.6), # SW corner
            (0.6, -0.6),  # SE corner
            (-0.6, 0.6)   # NW corner
        ]
        
        for i, (lane_name, (x, y, queue_len)) in enumerate(zip(lanes, positions)):
            # Draw queue bars
            if lane_name == 'N':
                bar_rect = Rectangle((x - 0.15, y - 0.3), 0.3, min(queue_len * 0.1, 0.8),
                                     fill=True, color='darkblue', alpha=0.7, zorder=2)
            elif lane_name == 'S':
                bar_rect = Rectangle((x - 0.15, y), 0.3, min(queue_len * 0.1, 0.8),
                                     fill=True, color='darkblue', alpha=0.7, zorder=2)
            elif lane_name == 'E':
                bar_rect = Rectangle((y, x - 0.15), min(queue_len * 0.1, 0.8), 0.3,
                                     fill=True, color='darkblue', alpha=0.7, zorder=2)
            else:  # W
                bar_rect = Rectangle((y - min(queue_len * 0.1, 0.8), x - 0.15), 
                                     min(queue_len * 0.1, 0.8), 0.3,
                                     fill=True, color='darkblue', alpha=0.7, zorder=2)
            ax.add_patch(bar_rect)
            
            # Draw queue length label
            ax.text(x, y, f'{lane_name}\n{queue_len}', ha='center', va='center',
                   fontsize=10, fontweight='bold', zorder=3)
        
        # Draw traffic lights
        for i, (light_x, light_y) in enumerate(light_positions):
            light_color = colors[i]
            circle = Circle((light_x, light_y), 0.15, color=light_color, zorder=4)
            ax.add_patch(circle)
        
        # Draw intersection center
        ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=True, color='white', zorder=2))
        ax.text(0, 0, 'INTERSECTION', ha='center', va='center', fontsize=8, zorder=3)
    
    def render_single(self, queues, phase, time_step, info=None):
        """
        Render a single intersection state.
        
        Args:
            queues: List of 4 queue lengths [N, S, E, W]
            phase: Current phase (0=NS Green, 1=EW Green)
            time_step: Current timestep
            info: Dictionary with additional info (throughput, total_wait, etc.)
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        else:
            self.ax.clear()
        
        self._draw_single_intersection(self.ax, queues, phase, "Traffic Intersection")
        
        # Add info text
        phase_str = "NS Green" if phase == 0 else "EW Green"
        info_text = f"Time: {time_step}\nPhase: {phase_str}"
        
        if info:
            info_text += f"\nThroughput: {info.get('throughput', 0)}"
            info_text += f"\nTotal Wait: {info.get('total_wait', 0):.1f}"
        
        self.fig.text(0.02, 0.98, info_text, transform=self.fig.transFigure,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return self.fig
    
    def render_multi(self, queues_1, phase_1, queues_2, phase_2, time_step, info=None):
        """
        Render two intersections side by side.
        
        Args:
            queues_1: List of 4 queue lengths for intersection 1
            phase_1: Current phase for intersection 1
            queues_2: List of 4 queue lengths for intersection 2
            phase_2: Current phase for intersection 2
            time_step: Current timestep
            info: Dictionary with additional info
        """
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=self.figsize)
        else:
            for ax in self.axes:
                ax.clear()
        
        self._draw_single_intersection(self.axes[0], queues_1, phase_1, "Intersection 1 (West)")
        self._draw_single_intersection(self.axes[1], queues_2, phase_2, "Intersection 2 (East)")
        
        # Add link visualization
        self.axes[0].text(1.8, 0, "→", fontsize=30, ha='center', va='center')
        
        # Add info text
        phase_str_1 = "NS Green" if phase_1 == 0 else "EW Green"
        phase_str_2 = "NS Green" if phase_2 == 0 else "EW Green"
        
        info_text = f"Time: {time_step}\n"
        info_text += f"Int1 Phase: {phase_str_1} | Int2 Phase: {phase_str_2}"
        
        if info:
            info_text += f"\nThroughput: {info.get('throughput', 0)}"
            info_text += f"\nTotal Wait: {info.get('total_wait', 0):.1f}"
        
        self.fig.text(0.5, 0.02, info_text, transform=self.fig.transFigure,
                     fontsize=10, ha='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        return self.fig
    
    def save_frame(self, filename):
        """Save current figure to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=100, bbox_inches='tight')
    
    def show(self):
        """Display the figure."""
        if self.fig is not None:
            plt.show()
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
