# Import the gym module
import gym
# make env before importing tensorflow, otherwise it will not load for some reason
env = gym.make('BreakoutDeterministic-v4')

import random
from atari_agent import AtariAgent
from atari_preprocessing import preprocess
import numpy as np
import os
from collections import deque

os.environ['KMP_DUPLICATE_LIB_OK']='True'

ATARI_SHAPE = (4, 105, 80)
BATCH_SIZE = 32
ACTIONS_SIZE = 4

actions = env.action_space
print(actions.n)
agent = AtariAgent(env, "test_model_new_2")

# agent.train(env)
agent.test(env)
