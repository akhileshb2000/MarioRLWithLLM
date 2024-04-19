import os

import gym
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from joypad_space_compatible import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

# Baseline reward - this function determines the reward at each step by calculating Mario’s velocity (positive points while moving right, negative points while moving left, zero while standing still),
# plus a penalty for every frame that passes to encourage movement, and a penalty if Mario dies for any reason.

# Human generated reward - this reward function rewards Mario for increasing his in-game score by defeating enemies, grabbing coins, and collecting power-ups.
# Additionally, a sizable reward is added if he collects the flag (or defeats Bowser) at the end of the level to encourage him to successfully beat the stage.
class HumanReward(gym.Wrapper):
    def __init__(self, env):
        super(HumanReward, self).__init__(env)
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
    
# LLM generated rewards (No edits)
# This reward function encourages the agent to complete the level as quickly as possible by rewarding it for decreasing the remaining time.
class TimeReward(gym.Wrapper):
    def __init__(self, env):
        super(TimeReward, self).__init__(env)
        self._current_time = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        time_reward = info['time'] - self._current_time
        self._current_time = info['time']
        reward += time_reward
        return state, reward, terminated, truncated, info

# This reward function encourages the agent to collect coins by rewarding it for increasing the coin count.
class CoinReward(gym.Wrapper):
    def __init__(self, env):
        super(CoinReward, self).__init__(env)
        self._current_coins = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        coin_reward = info['coins'] - self._current_coins
        self._current_coins = info['coins']
        reward += coin_reward * 5  # Adjust the coin reward scaling as needed
        return state, reward, terminated, truncated, info

# This reward function encourages the agent to stay high up on the screen, which can be beneficial for avoiding enemies and obstacles.
class HeightReward(gym.Wrapper):
    def __init__(self, env):
        super(HeightReward, self).__init__(env)
        self._prev_y_pos = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        height_reward = self._prev_y_pos - info['y_pos']
        self._prev_y_pos = info['y_pos']
        reward += height_reward
        return state, reward, terminated, truncated, info
    
# This reward function encourages the agent to move as far as possible to the right, which is the primary objective in Super Mario Bros.
class DistanceReward(gym.Wrapper):
    def __init__(self, env):
        super(DistanceReward, self).__init__(env)
        self._prev_x_pos = 0

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        distance_reward = info['x_pos'] - self._prev_x_pos
        self._prev_x_pos = info['x_pos']
        reward += distance_reward
        return state, reward, terminated, truncated, info

# This reward function rewards the agent for gaining power-ups (e.g., going from small to tall, or from tall to fireball) by increasing its status.
class PowerupReward(gym.Wrapper):
    def __init__(self, env):
        super(PowerupReward, self).__init__(env)
        self._prev_status = env._player_status

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if info['status'] > self._prev_status:
            reward += 50  # Adjust the reward value as needed
        self._prev_status = info['status']
        return state, reward, terminated, truncated, info

# This reward function rewards the agent for killing enemies by tracking the enemy types on the screen and rewarding the agent when the sum of enemy types decreases.
class EnemyKillReward(gym.Wrapper):
    def __init__(self, env):
        super(EnemyKillReward, self).__init__(env)
        self._prev_enemies = None

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        enemies = [self.env.ram[addr] for addr in self.env._ENEMY_TYPE_ADDRESSES]
        if self._prev_enemies is not None:
            enemy_diff = sum(self._prev_enemies) - sum(enemies)
            reward += enemy_diff * 10  # Adjust the reward value as needed
        self._prev_enemies = enemies
        return state, reward, terminated, truncated, info

# This reward function encourages the agent to explore new areas of the level by rewarding it for visiting new (x, y) positions on the screen.
class ExplorationReward(gym.Wrapper):
    def __init__(self, env):
        super(ExplorationReward, self).__init__(env)
        self._visited = set()

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        x_pos, y_pos = info['x_pos'], info['y_pos']
        pos = (x_pos, y_pos)
        if pos not in self._visited:
            self._visited.add(pos)
            reward += 1  # Adjust the reward value as needed
        return state, reward, terminated, truncated, info


def make_env(env_id: str, rank: int, seed: int = 0, render_mode: str = None):
    def _init():
        # 1. Create the base environment
        env = gym.make(env_id, apply_api_compatibility=True, render_mode=render_mode)
        # 2. Simplify the controls 
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        # 3. Grayscale
        env = GrayScaleObservation(env, keep_dim=True)
        # 4. Modify the reward function if needed
        env = HumanReward(env)

        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init
    
if __name__ == '__main__':
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    TOTAL_TIMESTEPS = 819200
    NUM_CPU = 4

    # 5. Create the vectorized environment
    vec_env = SubprocVecEnv([make_env("SuperMarioBros-v0", i) for i in range(NUM_CPU)])
    # 6. Stack the frames
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')


    # 7. Setup evaluation environment
    eval_env = DummyVecEnv([lambda: make_env("SuperMarioBros-v0", 0, render_mode="human")()])
    eval_env = VecFrameStack(eval_env, 4, channels_order='last')
    eval_env = VecTransposeImage(eval_env)
    eval_callback = EvalCallback(eval_env, best_model_save_path=CHECKPOINT_DIR, log_path=LOG_DIR, eval_freq=(TOTAL_TIMESTEPS / NUM_CPU / 4), deterministic=False, render=True, verbose=1, warn=False)

    # Train the AI model, this is where the AI model starts to learn
    model = PPO('CnnPolicy', vec_env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512, batch_size=256)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=eval_callback)

    # Save the AI model
    model.save('MarioRL')