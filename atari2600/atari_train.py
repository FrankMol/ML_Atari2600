# Import the gym module
import gym

# make env before importing tensorflow, otherwise it will not load for some reason
env = gym.make('BreakoutDeterministic-v4')  # training environment
env_test = gym.make('BreakoutDeterministic-v4')  # test environment

import numpy as np
import os
import sys
import random
import time
from atari_agent import AtariAgent
from atari_preprocessing import preprocess
from collections import deque
from tensorflow import flags
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK']='True'

default_model_name = "untitled_model_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
ATARI_SHAPE = (105, 80, 4)  # tensor flow backend -> channels last
FLAGS = flags.FLAGS

# define hyper parameters -> these can all be passed as command line arguments!
flags.DEFINE_boolean('use_checkpoints', True, "set if model will be saved during training. Set to False for debugging")
flags.DEFINE_integer('checkpoint_frequency', 1000, "number of iterations after which model file is updated")
flags.DEFINE_integer('max_iterations', 10000000, "number of iterations after which training is done")
flags.DEFINE_integer('batch_size', 32, "mini batch size")
flags.DEFINE_integer('memory_size', 1000000, "max number of stored states from which batch is sampled")
flags.DEFINE_integer('memory_start_size', 50000, "number of states with which the memory is initialized")
flags.DEFINE_integer('agent_history', 4, "number of frames in each state")
flags.DEFINE_float('initial_epsilon', 1, "initial value of epsilon used for exploration of state space")
flags.DEFINE_float('final_epsilon', 0.1, "final value of epsilon used for exploration of state space")
flags.DEFINE_integer('final_exploration_frame', 1000000, "frame at which final exploration reached")  # LET OP: frame/q?
flags.DEFINE_integer('no_op_max', 30, "max number of do nothing actions at beginning of episode")
flags.DEFINE_integer('update_frequency', 4, "number of actions played by agent between each q-iteration")
flags.DEFINE_integer('iteration', 0, "counter that keeps track of training iterations")


def get_epsilon_for_iteration(iteration):
    epsilon = max(FLAGS.final_epsilon, FLAGS.initial_epsilon - (FLAGS.initial_epsilon - FLAGS.final_epsilon)
                  / FLAGS.final_exploration_frame * iteration)
    return epsilon


def update_state(state, frame):
    frame = preprocess(frame)
    frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    next_state = np.append(frame, state[:, :, :, :3], axis=3)
    return next_state


def play_test_episode(agent):
    score = 0
    is_done = 0
    steps = 0
    frame = env_test.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    while not is_done and steps < 1000:
        frame, reward, is_done, _ = env_test.step(agent.choose_action(state, 0))
        state = update_state(state, frame)
        score += reward
        steps += 1
    if is_done:
        return score
    else:
        return "DNF"


# reset environment and get first state
def get_start_state():
    frame = env.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    return state


def initialize_memory(agent):
    # create memory object
    memory = deque(maxlen=FLAGS.memory_size)

    state = get_start_state()

    # choose epsilon for initialization of memory, use iteration provided by flags
    epsilon = get_epsilon_for_iteration(FLAGS.iteration)

    print("Initializing memory with {} states...".format(FLAGS.memory_start_size))
    no_op = random.randrange(FLAGS.no_op_max)
    for i in range(FLAGS.memory_start_size):
        if no_op > 0:
            action = 0
            no_op -= 1
        else:
            action = agent.choose_action(state, epsilon)

        frame, reward, is_done, _ = env.step(action)
        next_state = update_state(state, frame)
        memory.append((state, action, reward, next_state, is_done))
        state = next_state

        if is_done:
            env.reset()
            no_op = random.randrange(FLAGS.no_op_max)
    return memory


def main(argv):

    # make agent
    if len(argv) > 1:
        agent = AtariAgent(env, argv[1])
    else:
        agent = AtariAgent(env)

    # initialize replay memory
    memory = initialize_memory(agent)
    state = get_start_state()
    # start timer
    start_time = time.time()
    # start training
    iteration = FLAGS.iteration
    no_op = random.randrange(FLAGS.no_op_max) # 'do nothing' actions left to play
    try:
        while iteration < FLAGS.max_iterations:

            # let the agent play a number of steps in the game
            for i in range(FLAGS.update_frequency):
                # Choose epsilon based on the iteration
                epsilon = get_epsilon_for_iteration(iteration)

                # Choose the action -> do nothing at beginning of new episode
                if no_op > 0:
                    action = 0
                    no_op -= 1
                else:
                    action = agent.choose_action(state, epsilon)
                # Play one game iteration
                frame, reward, is_done, _ = env.step(action)
                next_state = update_state(state, frame)
                # add state to memory
                memory.append((state, action, reward, next_state, is_done))

                # reset game if game over
                if is_done:
                    state = get_start_state()
                    no_op = random.randrange(FLAGS.no_op_max)
                else:
                    state = next_state

            # Sample mini batch from memory and fit model
            batch = random.sample(memory, FLAGS.batch_size)
            agent.fit_batch(batch)

            # provide feedback about iteration, elapsed time, current performance
            if iteration % FLAGS.checkpoint_frequency == 0 and not iteration == FLAGS.iteration:
                score = play_test_episode(agent)
                cur_time = time.time()
                m, s = divmod(cur_time-start_time, 60)
                h, m = divmod(m, 60)
                time_str = "%d:%02d:%02d" % (h, m, s)
                print("iteration: {}, elapsed time: {}, score: {}".format(iteration, time_str, score))
                agent.save_model_to_file(iteration)
                # TODO: save separate model if score is best so far

            iteration += 1
    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    # save final state of model
    agent.save_model_to_file(iteration)

    env.close()
    env_test.close()


if __name__ == '__main__':
    main(sys.argv)
