from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env import RacingEnv

# Training environment
env = make_vec_env(RacingEnv, n_envs=4)
env = VecFrameStack(env, n_stack=4)

model = PPO(
    "CnnPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda",
    seed=42,
)

# Evaluation environment
eval_env = make_vec_env(RacingEnv, n_envs=1)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    log_path="./logs/",
    eval_freq=10000,
    n_eval_episodes=5,
    deterministic=True,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_car",
)

model.learn(
    total_timesteps=500000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
)

model.save("models/car_racing_v3")

env.close()
eval_env.close()