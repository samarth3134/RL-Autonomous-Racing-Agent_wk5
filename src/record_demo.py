from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from env import RacingEnv

env = DummyVecEnv([
    lambda: RecordVideo(
        RacingEnv(render_mode="rgb_array"),
        video_folder="./videos/",
        episode_trigger=lambda ep: True,
    )
])
env = VecFrameStack(env, n_stack=4)

model = PPO.load("models/best/best_model")

obs = env.reset()
episode_reward = 0
episode_count = 0

try:
    while episode_count < 5:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += rewards[0]

        if dones[0]:
            print(f"Episode {episode_count + 1} finished, reward: {episode_reward:.2f}")
            episode_reward = 0
            episode_count += 1
except KeyboardInterrupt:
    pass
finally:
    env.close()
