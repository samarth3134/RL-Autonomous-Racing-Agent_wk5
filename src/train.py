from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList

from env import RacingEnv


def describe_vec_env(vec_env, name: str) -> None:
    print(f"\n{name} wrapper chain:")
    current = vec_env
    depth = 0
    while True:
        print(f"  {depth}: {type(current).__name__}")
        if isinstance(current, VecEnvWrapper):
            current = current.venv
            depth += 1
            continue
        break


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class SteerJerkLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.prev_actions = None

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is None:
            return True

        if self.prev_actions is not None:
            jerk = abs(actions[:, 0] - self.prev_actions[:, 0])
            self.logger.record("custom/steer_jerk_mean", jerk.mean())

        self.prev_actions = actions.copy()
        return True


env = make_vec_env(RacingEnv, n_envs=4)
env = VecFrameStack(env, n_stack=4)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

model = PPO(
    "CnnPolicy",
    env,
    learning_rate=linear_schedule(3e-4),
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda",
    seed=42,
)

eval_env = make_vec_env(RacingEnv, n_envs=1)
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

# PPO adds VecTransposeImage around image observations for CnnPolicy. EvalCallback
# does not mirror that wrapper automatically, so keep both chains identical.
if isinstance(model.get_env(), VecTransposeImage):
    eval_env = VecTransposeImage(eval_env)

describe_vec_env(model.get_env(), "training env before EvalCallback")
describe_vec_env(eval_env, "eval env before EvalCallback")

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

steer_jerk_callback = SteerJerkLoggerCallback()

callbacks = CallbackList([eval_callback, checkpoint_callback, steer_jerk_callback])

describe_vec_env(model.get_env(), "training env before model.learn")
describe_vec_env(eval_callback.eval_env, "eval env before model.learn")

model.learn(
    total_timesteps=2_000_000,  # SMOKE TEST — bump to 2_000_000 once behavior looks right
    callback=callbacks,
    progress_bar=True,
)

model.save("models/car_racing_v3")

env.close()
eval_env.close()
