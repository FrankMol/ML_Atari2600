# Import the gym module
import gym
import sys
import time
from atari_preprocessing import preprocess

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

is_done = False
while True:
    # Perform random action, return new frame, reward and whether game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    if is_done:
        env.reset()
    # Render
    frame = preprocess(frame)
    time.sleep(0.1)
    env.render()
env.close()
