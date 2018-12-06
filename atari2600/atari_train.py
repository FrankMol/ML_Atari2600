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
from tensorflow import flags

os.environ['KMP_DUPLICATE_LIB_OK']='True'
FLAGS = flags.FLAGS
ATARI_SHAPE = (105, 80, 4) # tensorflow backend -> channels last

# create agent
agent = AtariAgent(env)

memory = deque(maxlen=FLAGS.memory_size)

def get_epsilon_for_iteration(iteration):
    epsilon = max(FLAGS.final_epsilon, FLAGS.initial_epsilon - (FLAGS.initial_epsilon - FLAGS.final_epsilon)
                  / FLAGS.final_exploration_frame * iteration)
    return epsilon


def update_state(state, frame):
    frame = preprocess(frame)
    frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    next_state = np.append(frame, state[:, :, :, :3], axis=3)
    return next_state


def initialize_memory(iteration):
    # get initial state
    frame = env.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))

    # choose epsilon for initialization of memory
    epsilon = get_epsilon_for_iteration(iteration)

    print("initializing memory with {} states...".format(FLAGS.memory_start_size))
    for i in range(FLAGS.memory_start_size):
        action = agent.choose_action(state, epsilon)

        frame, reward, is_done, _ = env.step(action)
        frame = preprocess(frame)
        frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
        next_state = np.append(frame, state[:, :, :, :3], axis=3)
        memory.append((state, action, reward, next_state, is_done))
        state = next_state

        if is_done:
            env.reset()
    return state


def __main__():

    iteration = 0
    state = initialize_memory(iteration)

    # start training
    while iteration < 10000000:

        # Choose epsilon based on the iteration
        epsilon = get_epsilon_for_iteration(iteration)

        # Choose the action
        action = agent.choose_action(state, epsilon)

        # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
        frame, reward, is_done, _ = env.step(action)
        next_state = update_state(state, frame)

        # add state to memory
        memory.append((state, action, reward, next_state, is_done))

        state = next_state

        if is_done:
            env.reset()

        # Sample and fit
        batch = random.sample(memory, FLAGS.batch_size)
        # unpack batch

        agent.fit_batch(batch)

        if iteration % 500 ==0:
            print("iteration: {}".format(iteration))

        if iteration % 1000 == 0:
            agent.save_model_to_file()

        iteration += 1

    env.close()


if __name__ == '__main__':
    __main__()
