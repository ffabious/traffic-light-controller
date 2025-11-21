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
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Ensure queues are integers
        queues = [int(q) for q in queues]
        
        # Background (Grass)
        ax.add_patch(Rectangle((-2.5, -2.5), 5, 5, fill=True, color='#4CAF50', zorder=0))
        
        # Road dimensions
        road_width = 1.0
        road_color = '#333333'
        marking_color = 'white'
        center_line_color = '#FDD835' # Yellow
        
        # Draw intersection roads (Cross shape)
        # Vertical road (NS)
        ax.add_patch(Rectangle((-road_width/2, -2.5), road_width, 5, fill=True, color=road_color, zorder=1))
        # Horizontal road (EW)
        ax.add_patch(Rectangle((-2.5, -road_width/2), 5, road_width, fill=True, color=road_color, zorder=1))
        
        # Draw intersection center
        ax.add_patch(Rectangle((-road_width/2, -road_width/2), road_width, road_width, fill=True, color='#444444', zorder=1))
        
        # Road Markings
        # Center lines (Dashed Yellow)
        # North leg
        ax.plot([0, 0], [0.5, 2.5], color=center_line_color, linestyle='--', linewidth=2, zorder=1)
        # South leg
        ax.plot([0, 0], [-0.5, -2.5], color=center_line_color, linestyle='--', linewidth=2, zorder=1)
        # East leg
        ax.plot([0.5, 2.5], [0, 0], color=center_line_color, linestyle='--', linewidth=2, zorder=1)
        # West leg
        ax.plot([-0.5, -2.5], [0, 0], color=center_line_color, linestyle='--', linewidth=2, zorder=1)
        
        # Stop lines (White solid)
        stop_line_width = 3
        # North 
        ax.plot([0, -0.5], [0.5, 0.5], color=marking_color, linewidth=stop_line_width, zorder=1)
        # South
        ax.plot([0, 0.5], [-0.5, -0.5], color=marking_color, linewidth=stop_line_width, zorder=1)
        # East
        ax.plot([0.5, 0.5], [0, 0.5], color=marking_color, linewidth=stop_line_width, zorder=1)
        # West
        ax.plot([-0.5, -0.5], [0, -0.5], color=marking_color, linewidth=stop_line_width, zorder=1)

        # Cars
        # Car dimensions
        car_length = 0.25
        car_width = 0.18
        car_gap = 0.05
        car_color = '#3F51B5' # Indigo
        
        # Queues: [N, S, E, W]
        # N Queue: North leg, Right lane (x > 0), facing South (down)
        # Starts at y=0.5 + gap, goes up
        for i in range(queues[0]):
            y_pos = 0.5 + 0.1 + i * (car_length + car_gap)
            if y_pos + car_length > 2.5: break # Don't draw off screen
            # Center of lane is x = 0.25
            rect = Rectangle((0.25 - car_width/2, y_pos), car_width, car_length, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)

        # S Queue: South leg, Right lane (x < 0), facing North (up)
        # Starts at y=-0.5 - gap, goes down
        for i in range(queues[1]):
            y_pos = -0.5 - 0.1 - car_length - i * (car_length + car_gap)
            if y_pos < -2.5: break
            # Center of lane is x = -0.25
            rect = Rectangle((-0.25 - car_width/2, y_pos), car_width, car_length, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)

        # E Queue: East leg, Right lane (y > 0), facing West (left)
        # Starts at x=0.5 + gap, goes right
        for i in range(queues[2]):
            x_pos = 0.5 + 0.1 + i * (car_length + car_gap)
            if x_pos + car_length > 2.5: break
            # Center of lane is y = 0.25
            # Car is horizontal
            rect = Rectangle((x_pos, 0.25 - car_width/2), car_length, car_width, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)

        # W Queue: West leg, Right lane (y < 0), facing East (right)
        # Starts at x=-0.5 - gap, goes left
        for i in range(queues[3]):
            x_pos = -0.5 - 0.1 - car_length - i * (car_length + car_gap)
            if x_pos < -2.5: break
            # Center of lane is y = -0.25
            rect = Rectangle((x_pos, -0.25 - car_width/2), car_length, car_width, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)

        # Traffic Lights
        # Phase 0: NS Green, EW Red
        # Phase 1: NS Red, EW Green
        ns_color = '#00E676' if phase == 0 else '#FF5252' # Green / Red
        ew_color = '#FF5252' if phase == 0 else '#00E676' # Red / Green
        
        light_radius = 0.1
        
        # NS Lights (Visible to N and S traffic)
        # Placed at corners
        for pos in [(-0.6, 0.6), (0.6, -0.6)]: # NW and SE corners
             # Box
            ax.add_patch(Rectangle((pos[0]-0.15, pos[1]-0.15), 0.3, 0.3, color='black', zorder=3))
            # Light
            ax.add_patch(Circle(pos, light_radius, color=ns_color, zorder=4))
            
        # EW Lights (Visible to E and W traffic)
        for pos in [(0.6, 0.6), (-0.6, -0.6)]: # NE and SW corners
             # Box
            ax.add_patch(Rectangle((pos[0]-0.15, pos[1]-0.15), 0.3, 0.3, color='black', zorder=3))
            # Light
            ax.add_patch(Circle(pos, light_radius, color=ew_color, zorder=4))
            
        # Add labels for queue lengths
        ax.text(0.25, 2.3, f"N: {queues[0]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        ax.text(-0.25, -2.3, f"S: {queues[1]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        ax.text(2.3, 0.25, f"E: {queues[2]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        ax.text(-2.3, -0.25, f"W: {queues[3]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
    
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
