from environment.single_env import SingleIntersectionEnv
from visual.visualizer import TrafficVisualizer

# Create environment
env = SingleIntersectionEnv()
obs, info = env.reset()

# Create visualizer
viz = TrafficVisualizer(env, mode='single', figsize=(10, 8))

# Run simulation and render
for step in range(50):
    action = 0 if step < 20 else 1  # Keep phase, then switch
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract queue lengths and phase from observation
    queues = obs[:4]  # [N, S, E, W]
    phase = int(obs[4])
    
    # Render
    viz.render_single(queues, phase, step, info)
    viz.show()  # Display in window
    # Or save: viz.save_frame(f'frame_{step:03d}.png')