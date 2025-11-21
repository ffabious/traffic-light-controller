from environment.multi_env import MultiIntersectionEnv
from visual.visualizer import TrafficVisualizer

env = MultiIntersectionEnv()
obs, info = env.reset()

viz = TrafficVisualizer(env, mode='multi', figsize=(14, 6))

for step in range(50):
    action = [0, 1]  # Different actions per intersection
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract queues and phases
    queues_1 = obs[:4]
    queues_2 = obs[4:8]
    phase_1 = int(obs[8])
    phase_2 = int(obs[9])
    
    # Render both intersections
    viz.render_multi(queues_1, phase_1, queues_2, phase_2, step, info)
    viz.show()