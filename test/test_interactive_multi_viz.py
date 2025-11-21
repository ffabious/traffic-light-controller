from environment.multi_env import MultiIntersectionEnv
from visual.visualizer import TrafficVisualizer
import random

def get_action(obs, info):
    # Random actions for both intersections
    return [random.choice([0, 1]), random.choice([0, 1])]

def main():
    # Create environment
    env = MultiIntersectionEnv()
    
    # Create visualizer
    viz = TrafficVisualizer(env, mode='multi', figsize=(14, 8))
    
    print("Starting interactive multi-intersection visualization...")
    print("Controls:")
    print("  Play: Start simulation")
    print("  Pause: Pause simulation")
    print("  Step: Advance one step")
    
    # Run interactive
    viz.interactive_run(get_action, max_steps=1000, interval=100)

if __name__ == "__main__":
    main()
