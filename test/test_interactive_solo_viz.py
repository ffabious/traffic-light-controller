from environment.traffic_env import TrafficIntersectionEnv
from visual.visualizer import TrafficVisualizer

def get_action(obs, info):
    # Simple policy: switch every 20 steps
    # We don't have access to step count here easily unless we track it or use info if available
    # But for random/simple testing:
    import random
    return random.choice([0, 1])

def main():
    # Create environment
    env = TrafficIntersectionEnv()
    
    # Create visualizer
    viz = TrafficVisualizer(env, mode='single', figsize=(10, 8))
    
    print("Starting interactive visualization...")
    print("Controls:")
    print("  Play: Start simulation")
    print("  Pause: Pause simulation")
    print("  Step: Advance one step")
    
    # Run interactive
    viz.interactive_run(get_action, max_steps=1000, interval=100)

if __name__ == "__main__":
    main()
