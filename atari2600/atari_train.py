# Import the gym module
import gym

# make env before importing tensorflow, otherwise it will not load for some reason
ENV_ID = "BreakoutDeterministic-v4"
env_train = gym.make(ENV_ID)  # training environment
env_test = gym.make(ENV_ID)  # evaluation environment

import numpy as np
import os
import sys
import time
import csv
from replay_memory import ReplayMemory
from atari_agent import AtariAgent
from atari_controller import AtariController
from tensorflow import flags
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

default_model_name = "untitled_model_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
ATARI_SHAPE = (80, 80, 4)  # tensor flow backend -> channels last
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


def evaluate_model(controller, agent, n_steps=FLAGS.eval_steps):
    episode_cnt = 0
    score = 0
    evaluation_score = 0
    controller.reset()
    no_op = True
    for _ in range(n_steps):
        if no_op:
            action = 1
            no_op = False
        else:
            action = agent.choose_action(controller.get_state(), FLAGS.eval_epsilon)
        _, reward, is_done, life_lost = controller.step(action)
        score += reward
        if is_done:
            evaluation_score += score
            score = 0
            controller.reset()
            episode_cnt += 1
            break
        if is_done or life_lost:
            no_op = True
    return evaluation_score/episode_cnt if episode_cnt > 0 else -1


def get_epsilon(iteration):
    epsilon = max(FLAGS.final_epsilon, FLAGS.initial_epsilon - (FLAGS.initial_epsilon - FLAGS.final_epsilon)
                  / FLAGS.annealing_steps * iteration)
    return epsilon


def write_logs(model_id, iteration, seconds, score):
    file_name = os.path.join(MODEL_PATH, model_id) + '.csv'
    with open(file_name, 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([iteration, round(seconds), score])


def main(argv):

    env = env_train
    # get model id from command line or use default name
    if len(argv) > 1 and not argv[1].startswith("--"):
        model_id = argv[1]
    else:
        model_id = default_model_name


    # instantiate agent, controller, memory
    controller = AtariController(env)
    evaluation_controller = AtariController(env_test)
    memory = ReplayMemory()
    agent = AtariAgent(env, model_id)

    # get start iteration (for resuming training on stored model)
    global_step = FLAGS.iteration - FLAGS.memory_start_size
    q_iteration = 0
    # start timer
    start_time = time.time()

    print("Initializing replay memory with {} states...".format(FLAGS.memory_start_size))

    # start training
    best_score = -np.inf  # keeps track of best score reached, used for saving best-so-far model
    try:
        while global_step < FLAGS.max_iterations:

            controller.reset()  # reset environment when done
            is_done = False

            while not is_done:
                # Choose the action -> do nothing at beginning of new episode
                action = agent.choose_action(controller.get_state(), get_epsilon(q_iteration))

                # interact with environment
                frame, reward, is_done, life_lost = controller.step(action)

                # add current experience to replay memory. both game over and dead are considered terminal states
                terminal = is_done or life_lost
                memory.add_experience(action, frame, reward, terminal)

                # Sample mini batch from memory and fit model
                if global_step % FLAGS.update_frequency == 0 and global_step > 0:
                    batch = memory.get_minibatch()
                    agent.fit_batch(batch)
                    q_iteration += 1

                    # provide feedback about iteration, elapsed time, current performance
                    if q_iteration == 1:
                        start_time = time.time()

                    if q_iteration % FLAGS.checkpoint_frequency == 0 and global_step > 0:
                        score = evaluate_model(evaluation_controller, agent)  # play evaluation episode to rate performance
                        cur_time = time.time()
                        m, s = divmod(cur_time-start_time, 60)
                        h, m = divmod(m, 60)
                        time_str = "%d:%02d:%02d" % (h, m, s)
                        # check if score is best so far and update model file(s)
                        is_highest = score > best_score
                        if FLAGS.use_checkpoints:
                            agent.save_checkpoint(q_iteration, is_highest)
                        if is_highest:
                            best_score = score
                        print("iteration {}, elapsed time: {}, score: {}, best: {}".format(q_iteration, time_str,
                                                                                           round(score, 2),
                                                                                           round(best_score, 2)))
                        write_logs(model_id, q_iteration, cur_time-start_time, score)

                global_step += 1

    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    # save final state of model
    if FLAGS.use_checkpoints:
        agent.save_checkpoint(q_iteration)
        print("Latest checkpoint at iteration {}".format(q_iteration))

    env.close()
    env_test.close()


if __name__ == '__main__':
    main(sys.argv)
