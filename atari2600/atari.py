# Import the gym module
import gym
# make env before importing tensorflow, otherwise it will not load for some reason
env = gym.make('BreakoutDeterministic-v4')

from atari_agent import AtariAgent
from atari_preprocessing import preprocess
import numpy as np
import os
import time
from tensorflow import flags

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ATARI_SHAPE = (105, 80, 4) # tensorflow backend -> channels last
FLAGS = flags.FLAGS

# make agent
agent = AtariAgent(env)

def update_state(state, frame):
    frame = preprocess(frame)
    frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    next_state = np.append(frame, state[:, :, :, :3], axis=3)
    return next_state


def test():
    score = 0
    is_done = 0
    steps = 0
    frame = env.reset()
    env.render()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    while not is_done and steps<1000:
        frame, reward, is_done, _ = env.step(agent.choose_action(state, 0))
        env.render()
        state = update_state(state, frame)
        score += reward
        steps += 1
    ##### CONTINUE HERE ######


if __name__ == "__main__":
    test()