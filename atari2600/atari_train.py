# Import the gym module
import gym

# make env before importing tensorflow, otherwise it will not load for some reason
env_train = gym.make('PongDeterministic-v4')  # training environment
env_test = gym.make('PongDeterministic-v4')  # test environment

import numpy as np
import os
import sys
import random
import time
import csv
from atari_agent import AtariAgent
from atari_preprocessing import preprocess
from collections import deque
from tensorflow import flags
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

default_model_name = "untitled_model_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
ATARI_SHAPE = (84, 84, 4)  # tensor flow backend -> channels last
FLAGS = flags.FLAGS
MODEL_PATH = 'trained_models/'

# define hyper parameters -> these can all be passed as command line arguments!
flags.DEFINE_boolean('use_checkpoints', True, "set if model will be saved during training. Set to False for debugging")
flags.DEFINE_integer('checkpoint_frequency', 10000, "number of iterations after which model file is updated")
flags.DEFINE_integer('max_iterations', 20000000, "number of iterations after which training is done")
flags.DEFINE_integer('batch_size', 32, "mini batch size")
flags.DEFINE_integer('memory_size', 1000000, "max number of stored states from which batch is sampled")
flags.DEFINE_integer('memory_start_size', 50000, "number of states with which the memory is initialized")
flags.DEFINE_integer('agent_history', 4, "number of frames in each state")
flags.DEFINE_float('initial_epsilon', 1, "initial value of epsilon used for exploration of state space")
flags.DEFINE_float('final_epsilon', 0.1, "final value of epsilon used for exploration of state space")
flags.DEFINE_float('eval_epsilon', 0.0, "value of epsilon used in epsilon-greedy policy evaluation")
flags.DEFINE_integer('eval_steps', 10000, "number of evaluation steps used to evaluate performance")
flags.DEFINE_integer('annealing_steps', 1000000, "frame at which final exploration reached")  # LET OP: frame/q?
flags.DEFINE_integer('no_op_max', 10, "max number of do nothing actions at beginning of episode")
flags.DEFINE_integer('no_op_action', 0, "action that the agent plays as no-op at beginning of episode")
flags.DEFINE_integer('update_frequency', 4, "number of actions played by agent between each q-iteration")
flags.DEFINE_integer('iteration', 0, "iteration at which training should start or resume")


def get_epsilon_for_iteration(iteration):
    epsilon = max(FLAGS.final_epsilon, FLAGS.initial_epsilon - (FLAGS.initial_epsilon - FLAGS.final_epsilon)
                  / FLAGS.annealing_steps * iteration)
    return epsilon


def update_state(state, frame):
    frame = preprocess(frame)
    frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    next_state = np.append(frame, state[:, :, :, :3], axis=3)
    return next_state


# reset environment and get first state
def get_start_state(env):
    frame = env.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    return state


def evaluate_model(env, agent, n_steps=FLAGS.eval_steps):
    episode_cnt = 0
    score = 0
    evaluation_score = 0
    no_op = random.randrange(FLAGS.no_op_max + 1)  # also use no-op in evaluation!
    state = get_start_state(env)
    for _ in range(n_steps):
        if no_op > 0:
            action = FLAGS.no_op_action if FLAGS.no_op_action >= 0 else env.action_space.sample()
            no_op -= 1
        else:
            action = agent.choose_action(state, FLAGS.eval_epsilon)
        frame, reward, is_done, _ = env.step(action)
        state = update_state(state, frame)
        score += reward
        if is_done:
            evaluation_score += score
            score = 0
            state = get_start_state(env)
            episode_cnt += 1
    return evaluation_score/episode_cnt if episode_cnt > 0 else -1


def write_logs(model_id, iteration, seconds, score):
    file_name = os.path.join(MODEL_PATH, model_id) + '.csv'
    with open(file_name, 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([iteration, round(seconds), score])


# create a replay memory in the form of a deque, and fill with a number of states
def initialize_memory(env, agent):
    # create memory object
    memory = deque(maxlen=FLAGS.memory_size)

    state = get_start_state(env)

    # choose epsilon for initialization of memory, use iteration provided by flags
    epsilon = get_epsilon_for_iteration(FLAGS.iteration)

    print("Initializing replay memory with {} states...".format(FLAGS.memory_start_size))
    no_op = random.randrange(FLAGS.no_op_max+1)  # add 1 so that no_op_max can be set to 0
    for i in range(FLAGS.memory_start_size):
        if no_op > 0:
            action = FLAGS.no_op_action if FLAGS.no_op_action >= 0 else env.action_space.sample()
            no_op -= 1
        else:
            action = agent.choose_action(state, epsilon)

        frame, reward, is_done, _ = env.step(action)
        # clip reward
        reward = np.sign(reward)
        next_state = update_state(state, frame)
        memory.append((state, action, reward, next_state, is_done))
        state = next_state

        if is_done:
            env.reset()
            no_op = random.randrange(FLAGS.no_op_max+1)  # add 1 so that no_op_max can be set to 0
    print("Replay memory initialized")
    return memory


def main(argv):

    env = env_train
    # get model id from command line or use default name
    if len(argv) > 1 and not argv[1].startswith("--"):
        model_id = argv[1]
    else:
        model_id = default_model_name

    # instantiate agent
    agent = AtariAgent(env, model_id)

    # initialize replay memory and state
    memory = initialize_memory(env, agent)
    state = get_start_state(env)

    iteration = FLAGS.iteration

    # start timer
    start_time = time.time()
    if iteration == 0:
        print("Starting training...")
    else:
        print("Resuming training at iteration {}...".format(iteration))

    # start training
    no_op = random.randrange(FLAGS.no_op_max+1)  # add 1 so that no_op_max can be set to 0
    best_score = -np.inf  # keeps track of best score reached, used for saving best-so-far model
    try:
        while iteration < FLAGS.max_iterations:

            # let the agent play a number of steps in the game
            for i in range(FLAGS.update_frequency):
                # Choose epsilon based on the iteration
                epsilon = get_epsilon_for_iteration(iteration)

                # Choose the action -> do nothing at beginning of new episode
                if no_op > 0:
                    action = FLAGS.no_op_action if FLAGS.no_op_action >= 0 else env.action_space.sample()
                    no_op -= 1
                else:
                    action = agent.choose_action(state, epsilon)
                # Play one game iteration
                frame, reward, is_done, _ = env.step(action)
                # clip reward
                reward = np.sign(reward)
                # update state by adding new frame and removing oldest frame
                next_state = update_state(state, frame)
                # add state to memory
                memory.append((state, action, reward, next_state, is_done))

                # reset game if game over
                if is_done:
                    state = get_start_state(env)
                    no_op = random.randrange(FLAGS.no_op_max+1)  # add 1 so that no_op_max can be set to 0
                else:
                    state = next_state

            # Sample mini batch from memory and fit model
            batch = random.sample(memory, FLAGS.batch_size)
            agent.fit_batch(batch)

            # provide feedback about iteration, elapsed time, current performance
            if iteration % FLAGS.checkpoint_frequency == 0 and not iteration == FLAGS.iteration:
                score = evaluate_model(env, agent)  # play complete test episode to rate performance
                cur_time = time.time()
                m, s = divmod(cur_time-start_time, 60)
                h, m = divmod(m, 60)
                time_str = "%d:%02d:%02d" % (h, m, s)
                # check if score is best so far and update model file(s)
                is_highest = score > best_score
                agent.save_checkpoint(iteration, is_highest)
                if is_highest:
                    best_score = score
                print("iteration {}, elapsed time: {}, score: {}, best: {}".format(iteration, time_str, round(score, 2),
                                                                                   round(best_score, 2)))
                write_logs(model_id, iteration, cur_time-start_time, score)

            iteration += 1
    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    # save final state of model
    agent.save_checkpoint(iteration)
    print("Latest checkpoint at iteration {}".format(iteration))

    env.close()
    env_test.close()


if __name__ == '__main__':
    main(sys.argv)
