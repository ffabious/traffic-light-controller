import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
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
    
    def _draw_single_intersection(self, ax, queues, phase, title="Intersection", yellow_timer=0):
        """
        Draw a single intersection with queue visualization.
        
        Args:
            ax: Matplotlib axis
            queues: List of 4 queue lengths [N, S, E, W] - can be ints or deques
            phase: Current phase (0=NS Green, 1=EW Green)
            yellow_timer: Remaining time for yellow light (0 if not active)
        """
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Handle both numeric and deque queue representations
        queue_lengths = []
        queue_objects = []
        for q in queues:
            if isinstance(q, (int, float, np.integer, np.floating)):
                queue_lengths.append(int(q))
                queue_objects.append(None)
            else:
                # It's a deque or list of vehicles
                queue_lengths.append(len(q))
                queue_objects.append(list(q))
        
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
        # Helper to get length and vehicle ID
        def get_queue_info(q_data):
            if isinstance(q_data, (int, float, np.integer, np.floating)):
                return int(q_data), None
            else:
                # Assume it's a list/deque of vehicles
                return len(q_data), list(q_data)

        # N Queue: North leg, Right lane (x > 0), facing South (down)
        # Starts at y=0.5 + gap, goes up
        q_len, q_vehs = get_queue_info(queue_lengths[0]) if queue_objects[0] is None else (queue_lengths[0], queue_objects[0])
        for i in range(q_len):
            y_pos = 0.5 + 0.1 + i * (car_length + car_gap)
            if y_pos + car_length > 2.5: break # Don't draw off screen
            # Center of lane is x = -0.25 (West side for RHT)
            rect = Rectangle((-0.25 - car_width/2, y_pos), car_width, car_length, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)
            if q_vehs:
                veh_id = q_vehs[i].id
                ax.text(-0.25, y_pos + car_length/2, str(veh_id), 
                        ha='center', va='center', color='white', fontsize=6, zorder=3)

        # S Queue: South leg, Right lane (x < 0), facing North (up)
        # Starts at y=-0.5 - gap, goes down
        q_len, q_vehs = get_queue_info(queue_lengths[1]) if queue_objects[1] is None else (queue_lengths[1], queue_objects[1])
        for i in range(q_len):
            y_pos = -0.5 - 0.1 - car_length - i * (car_length + car_gap)
            if y_pos < -2.5: break
            # Center of lane is x = 0.25 (East side for RHT)
            rect = Rectangle((0.25 - car_width/2, y_pos), car_width, car_length, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)
            if q_vehs:
                veh_id = q_vehs[i].id
                ax.text(0.25, y_pos + car_length/2, str(veh_id), 
                        ha='center', va='center', color='white', fontsize=6, zorder=3)

        # E Queue: East leg, Right lane (y > 0), facing West (left)
        # Starts at x=0.5 + gap, goes right
        q_len, q_vehs = get_queue_info(queue_lengths[2]) if queue_objects[2] is None else (queue_lengths[2], queue_objects[2])
        for i in range(q_len):
            x_pos = 0.5 + 0.1 + i * (car_length + car_gap)
            if x_pos + car_length > 2.5: break
            # Center of lane is y = 0.25
            # Car is horizontal
            rect = Rectangle((x_pos, 0.25 - car_width/2), car_length, car_width, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)
            if q_vehs:
                veh_id = q_vehs[i].id
                ax.text(x_pos + car_length/2, 0.25, str(veh_id), 
                        ha='center', va='center', color='white', fontsize=6, zorder=3)

        # W Queue: West leg, Right lane (y < 0), facing East (right)
        # Starts at x=-0.5 - gap, goes left
        q_len, q_vehs = get_queue_info(queue_lengths[3]) if queue_objects[3] is None else (queue_lengths[3], queue_objects[3])
        for i in range(q_len):
            x_pos = -0.5 - 0.1 - car_length - i * (car_length + car_gap)
            if x_pos < -2.5: break
            # Center of lane is y = -0.25
            rect = Rectangle((x_pos, -0.25 - car_width/2), car_length, car_width, 
                             fill=True, color=car_color, ec='black', zorder=2)
            ax.add_patch(rect)
            if q_vehs:
                veh_id = q_vehs[i].id
                ax.text(x_pos + car_length/2, -0.25, str(veh_id), 
                        ha='center', va='center', color='white', fontsize=6, zorder=3)

        # Traffic Lights
        # Phase 0: NS Green, EW Red
        # Phase 1: NS Red, EW Green
        ns_color = '#00E676' if phase == 0 else '#FF5252' # Green / Red
        ew_color = '#FF5252' if phase == 0 else '#00E676' # Red / Green
        
        if yellow_timer > 0:
            # Transitioning TO 'phase'
            # If phase is 0 (NS Green), it means we came from EW Green. So EW is Yellow, NS Red.
            if phase == 0: 
                ns_color = '#FF5252' # Red
                ew_color = '#FFEB3B' # Yellow
            # If phase is 1 (EW Green), it means we came from NS Green. So NS is Yellow, EW Red.
            else: 
                ns_color = '#FFEB3B' # Yellow
                ew_color = '#FF5252' # Red
        
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
        ax.text(-0.25, 2.3, f"N: {queue_lengths[0]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        ax.text(0.25, -2.3, f"S: {queue_lengths[1]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        ax.text(2.3, 0.25, f"E: {queue_lengths[2]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        ax.text(-2.3, -0.25, f"W: {queue_lengths[3]}", ha='center', va='center', color='white', fontweight='bold', zorder=5, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
    
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
        
        # Use actual vehicles if available
        if hasattr(self.env, 'lanes') and len(self.env.lanes) >= 4:
            queues = [self.env.lanes[i].queue for i in range(4)]

        yellow_timer = 0
        if hasattr(self.env, 'yellow_timer'):
            yellow_timer = self.env.yellow_timer
        elif info and 'yellow_timer' in info:
            yellow_timer = info['yellow_timer']

        self._draw_single_intersection(self.ax, queues, phase, "Traffic Intersection", yellow_timer=yellow_timer)
        
        # Add info text
        phase_str = "NS Green" if phase == 0 else "EW Green"
        if yellow_timer > 0:
            phase_str += f" (Yellow: {yellow_timer})"
            
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
        
        # Use actual vehicles if available
        if hasattr(self.env, 'lanes') and len(self.env.lanes) >= 8:
            queues_1 = [self.env.lanes[i].queue for i in range(4)]
            queues_2 = [self.env.lanes[i].queue for i in range(4, 8)]

        yt1, yt2 = 0, 0
        if hasattr(self.env, 'yellow_timer'):
            if isinstance(self.env.yellow_timer, list):
                yt1, yt2 = self.env.yellow_timer
            else:
                yt1 = self.env.yellow_timer # Fallback

        self._draw_single_intersection(self.axes[0], queues_1, phase_1, "Intersection 1 (West)", yellow_timer=yt1)
        self._draw_single_intersection(self.axes[1], queues_2, phase_2, "Intersection 2 (East)", yellow_timer=yt2)
        
        # Add info text
        phase_str_1 = "NS Green" if phase_1 == 0 else "EW Green"
        if yt1 > 0: phase_str_1 += f" (Y: {yt1})"
        
        phase_str_2 = "NS Green" if phase_2 == 0 else "EW Green"
        if yt2 > 0: phase_str_2 += f" (Y: {yt2})"
        
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
    
    def interactive_run(self, get_action_callback, max_steps=1000, interval=200):
        """
        Run the simulation interactively with buttons.
        
        Args:
            get_action_callback: Function that takes (obs, info) and returns action
            max_steps: Maximum number of steps to run
            interval: Time between frames in ms
        """
        # Setup figure if not exists
        if self.fig is None:
            if self.mode == 'single':
                self.fig, self.ax = plt.subplots(figsize=self.figsize)
            else:
                self.fig, self.axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Add space for buttons at the bottom
        plt.subplots_adjust(bottom=0.2)
        
        # State
        self.running = False
        self.current_step = 0
        self.max_steps = max_steps
        self.obs, self.info = self.env.reset()
        self.get_action = get_action_callback
        
        # Buttons
        ax_play = plt.axes([0.59, 0.05, 0.1, 0.075])
        ax_pause = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_step = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_quit = plt.axes([0.92, 0.05, 0.07, 0.075])
        
        self.btn_play = Button(ax_play, 'Play')
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_step = Button(ax_step, 'Step')
        self.btn_quit = Button(ax_quit, 'Quit')
        
        def play(event):
            self.running = True
            
        def pause(event):
            self.running = False
            
        def step(event):
            self.running = False
            self.update_frame(None)
            plt.draw()
            
        def quit_sim(event):
            self.running = False
            self.current_step = self.max_steps
            plt.close(self.fig)
            
        self.btn_play.on_clicked(play)
        self.btn_pause.on_clicked(pause)
        self.btn_step.on_clicked(step)
        self.btn_quit.on_clicked(quit_sim)
        
        # Initial draw
        self.update_frame('init')
        
        # Animation
        self.anim = FuncAnimation(self.fig, self.update_frame, frames=range(max_steps), 
                                  interval=interval, repeat=False, cache_frame_data=False)
        plt.show()
        
    def update_frame(self, frame):
        if frame == 'init':
            # Just render current state (initial)
            pass
        elif not self.running and frame is not None:
            return
        elif self.current_step >= self.max_steps:
            self.running = False
            return
        else:
            # Step environment (if running or manual step)
            # Get action
            action = self.get_action(self.obs, self.info)
            
            # Step environment
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            self.current_step += 1
            
            if terminated or truncated:
                self.running = False
                self.obs, self.info = self.env.reset()
                self.current_step = 0

        # Render
        if self.mode == 'single':
            queues = self.obs[:4]
            phase = int(self.obs[4])
            
            # Use actual vehicles if available
            if hasattr(self.env, 'lanes') and len(self.env.lanes) >= 4:
                queues = [self.env.lanes[i].queue for i in range(4)]
            
            yellow_timer = 0
            if hasattr(self.env, 'yellow_timer'):
                yellow_timer = self.env.yellow_timer
            
            self.ax.clear()
            self._draw_single_intersection(self.ax, queues, phase, "Traffic Intersection", yellow_timer=yellow_timer)
            
            phase_str = "NS Green" if phase == 0 else "EW Green"
            if yellow_timer > 0: phase_str += f" (Y: {yellow_timer})"
            
            info_text = f"Time: {self.current_step}\nPhase: {phase_str}"
            if self.info:
                info_text += f"\nThroughput: {self.info.get('throughput', 0)}"
                info_text += f"\nTotal Wait: {self.info.get('total_wait', 0):.1f}"
            
            # Using ax.text for info to keep it simple and cleared with ax.clear()
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        else:
            queues_1 = self.obs[:4]
            queues_2 = self.obs[4:8]
            phase_1 = int(self.obs[8])
            phase_2 = int(self.obs[9])
            
            # Use actual vehicles if available
            if hasattr(self.env, 'lanes') and len(self.env.lanes) >= 8:
                queues_1 = [self.env.lanes[i].queue for i in range(4)]
                queues_2 = [self.env.lanes[i].queue for i in range(4, 8)]
            
            yt1, yt2 = 0, 0
            if hasattr(self.env, 'yellow_timer'):
                if isinstance(self.env.yellow_timer, list):
                    yt1, yt2 = self.env.yellow_timer
                else:
                    yt1 = self.env.yellow_timer

            for ax in self.axes:
                ax.clear()
            
            self._draw_single_intersection(self.axes[0], queues_1, phase_1, "Intersection 1 (West)", yellow_timer=yt1)
            self._draw_single_intersection(self.axes[1], queues_2, phase_2, "Intersection 2 (East)", yellow_timer=yt2)
            
            phase_str_1 = "NS Green" if phase_1 == 0 else "EW Green"
            if yt1 > 0: phase_str_1 += f" (Y: {yt1})"
            
            phase_str_2 = "NS Green" if phase_2 == 0 else "EW Green"
            if yt2 > 0: phase_str_2 += f" (Y: {yt2})"
            
            info_text = f"Time: {self.current_step}\n"
            info_text += f"Int1 Phase: {phase_str_1} | Int2 Phase: {phase_str_2}"
            
            if self.info:
                info_text += f"\nThroughput: {self.info.get('throughput', 0)}"
                info_text += f"\nTotal Wait: {self.info.get('total_wait', 0):.1f}"
                
            # Use figure text but we need to clear it or update it.
            # Easier to use a fixed text artist.
            if not hasattr(self, 'info_text_artist'):
                self.info_text_artist = self.fig.text(0.5, 0.25, info_text, transform=self.fig.transFigure,
                            fontsize=10, ha='center',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                self.info_text_artist.set_text(info_text)

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
