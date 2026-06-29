from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from env import RacingEnv

env = make_vec_env(
    lambda: RacingEnv(render_mode="human"),
    n_envs=1,
)

env = VecFrameStack(env, n_stack=4)

model = PPO.load("models/best/best_model")

obs = env.reset()

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(action)

        if dones[0]:
            obs = env.reset()

except KeyboardInterrupt:
    pass

finally:
    env.close()