import os 
from matplotlib import pyplot as plt

import gym
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# 1. Create the base environment
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
# 2. Simplify the controls 
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

model = PPO.load('./train/best_model_130000') # Replace with the path to the model you want to test

# Start the game 
state = env.reset()

# Loop through the game
while True: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()