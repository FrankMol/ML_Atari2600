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

actions = env.action_space
agent = AtariAgent(env)


# agent.train(env)
# agent.test(env)
