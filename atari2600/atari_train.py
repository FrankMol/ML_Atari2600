# Import the gym module
import gym
# make env before importing tensorflow, otherwise it will not load for some reason
env = gym.make('BreakoutDeterministic-v4') # training environment
env_test = gym.make('BreakoutDeterministic-v4') # test environment

import random
from atari_agent import AtariAgent
from atari_preprocessing import preprocess
import numpy as np
import os
import time
from datetime import datetime
from collections import deque
from tensorflow import flags

os.environ['KMP_DUPLICATE_LIB_OK']='True'

FLAGS = flags.FLAGS

# define hyperparameters -> these can all be passed as command line arguments!
flags.DEFINE_boolean('use_checkpoints', True, "set if model will be saved during training. Set to False for debugging")
flags.DEFINE_integer('checkpoint_frequency', 1000, "number of iterations after which model file is updated")
flags.DEFINE_integer('max_iterations', 10000000, "number of iterations after which training is done")
flags.DEFINE_integer('batch_size', 32, "mini batch size")
flags.DEFINE_integer('memory_size', 1000000, "max number of stored states from which batch is sampled")
flags.DEFINE_integer('memory_start_size', 50000, "number of states with which the memory is initialized")
flags.DEFINE_integer('agent_history', 4, "number of frames in each state")
flags.DEFINE_float('initial_epsilon', 1, "initial value of epsilon used for exploration of state space")
flags.DEFINE_float('final_epsilon', 0.1, "final value of epsilon used for exploration of state space")
flags.DEFINE_integer('final_exploration_frame', 1000000,
                   "frame at which final exploration reached") # LET OP: frame of q-iteration? -> for now FRAME!
flags.DEFINE_integer('no_op_max', 30, "max number of do nothing actions at beginning of episode")
flags.DEFINE_integer('update_frequency', 4, "number of actions played by agent between each q-iteration")
flags.DEFINE_integer('iteration', 0, "counter that keeps track of training iterations")

ATARI_SHAPE = (105, 80, 4) # tensorflow backend -> channels last

# create agent
agent = AtariAgent(env)
# create memory
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


def play_test_episode():
    score = 0
    is_done = 0
    steps = 0
    frame = env_test.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    while not is_done and steps<1000:
        frame, reward, is_done, _ = env_test.step(agent.choose_action(state, 0))
        state = update_state(state, frame)
        score += reward
        steps += 1
    if is_done:
        return score
    else:
        return "DNF"


def initialize_memory():
    # get initial state
    frame = env.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))

    # choose epsilon for initialization of memory, use iteration provided by flags
    epsilon = get_epsilon_for_iteration(FLAGS.iteration)

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


def train():

    iteration = FLAGS.iteration
    state = initialize_memory()

    start_time = time.time()
    # start training
    try:
        while iteration < 10000000:

            # Choose epsilon based on the iteration
            epsilon = get_epsilon_for_iteration(iteration)
            # Choose the action
            action = agent.choose_action(state, epsilon)
            # Play one game iteration
            frame, reward, is_done, _ = env.step(action)
            next_state = update_state(state, frame)
            # add state to memory
            memory.append((state, action, reward, next_state, is_done))
            state = next_state

            # reset game if game over
            if is_done:
                env.reset()

            # Sample and fit
            if iteration % FLAGS.update_frequency == 0:
                batch = random.sample(memory, FLAGS.batch_size)
                agent.fit_batch(batch)

            # provide feedback about iteration, elapsed time, current performance
            if iteration % FLAGS.checkpoint_frequency == 0 and not iteration == FLAGS.iteration:
                score = play_test_episode()
                cur_time = time.time()
                m, s = divmod(cur_time-start_time, 60)
                h, m = divmod(m, 60)
                timestr = "%d:%02d:%02d" % (h, m, s)
                print("iteration: {}, elapsed time: {}, score: {}".format(iteration, timestr, score))
                agent.save_model_to_file(iteration)

            iteration += 1
    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    # save final state of model
    agent.save_model_to_file(iteration)

    env.close()
    env_test.close()


if __name__ == '__main__':
    train()
