import os

import gym
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

_vec_reset = VecFrameStack.reset
_dummy_reset = DummyVecEnv.reset

def vec_reset(*args, **kwargs):
    return _vec_reset(*args)

def dummy_reset(*args, **kwargs):
    return _dummy_reset(*args)

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
VecFrameStack.reset = vec_reset
DummyVecEnv.reset = dummy_reset

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if terminated or truncated:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, terminated, truncated, info

def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        # 1. Create the base environment
        env = gym.make(env_id, apply_api_compatibility=True, render_mode="human")
        # 2. Modify the reward function
        env = CustomReward(env)
        # 3. Simplify the controls 
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        # 4. Grayscale
        env = GrayScaleObservation(env, keep_dim=True)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    
if __name__ == '__main__':
    CHECKPOINT_DIR = './train2/'
    LOG_DIR = './logs/'
    NUM_CPU = 2  # Number of processes to use

    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env("SuperMarioBros-v0", i) for i in range(NUM_CPU)])
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')

    # Setup model saving callback
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    # This is the AI model started
    model = PPO('CnnPolicy', vec_env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512, batch_size=256)

    # Train the AI model, this is where the AI model starts to learn
    model.learn(total_timesteps=1000000, callback=callback)

    # Save the AI model
    model.save('MarioRL')