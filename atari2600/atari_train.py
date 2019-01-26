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
import random
import tensorflow as tf
from replay_memory import ReplayMemory
from atari_agent import AtariAgent
from atari_controller import AtariController
from tensorflow import flags
from datetime import datetime
from gif_maker import convert_tensor_to_gif_summary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # blocks irrelevant tensorflow warning

default_model_name = "untitled_model_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
ATARI_SHAPE = (105, 80, 4)  # tensorflow backend -> channels last
FLAGS = flags.FLAGS
MODEL_PATH = 'trained_models/'
os.makedirs(MODEL_PATH, exist_ok=True)

SUMMARIES = "summaries/"          # logdir for tensorboard
RUNID = sys.argv[1]
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

# tensorboard
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    SCORE_PH = tf.placeholder(tf.float32, shape=None, name='score_summary')
    SCORE_SUMMARY = tf.summary.scalar('score', SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, SCORE_SUMMARY])

# GIF_PH = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='gif_summary')
# GIF_SUMMARY = tf.summary.tensor_summary('gif', GIF_PH)

# define hyper parameters -> these can all be passed as command line arguments!
flags.DEFINE_boolean('use_checkpoints', True, "set if model will be saved during training. Set to False for debugging")
flags.DEFINE_integer('checkpoint_frequency', 50000, "number of iterations after which model file is updated")
flags.DEFINE_integer('max_iterations', 20000000, "number of iterations after which training is done")
flags.DEFINE_integer('batch_size', 32, "mini batch size")
flags.DEFINE_integer('memory_size', 1000000, "max number of stored states from which batch is sampled")
flags.DEFINE_integer('memory_start_size', 50000, "number of states with which the memory is initialized")
flags.DEFINE_integer('agent_history', 4, "number of frames in each state")
flags.DEFINE_float('initial_epsilon', 1, "initial value of epsilon used for exploration of state space")
flags.DEFINE_float('final_epsilon', 0.1, "final value of epsilon used for exploration of state space")
flags.DEFINE_float('eval_epsilon', 0.05, "value of epsilon used in epsilon-greedy policy evaluation")
flags.DEFINE_integer('eval_steps', 10000, "number of evaluation steps used to evaluate performance")
flags.DEFINE_integer('eval_episodes', 10, "number of evaluation steps used to evaluate performance")
flags.DEFINE_integer('annealing_steps', 1000000, "frame at which final exploration reached")  # LET OP: frame/q?
flags.DEFINE_integer('no_op_max', 10, "max number of do nothing actions at beginning of episode")
flags.DEFINE_integer('no_op_action', 0, "action that the agent plays as no-op at beginning of episode")
flags.DEFINE_integer('update_frequency', 4, "number of actions played by agent between each q-iteration")
flags.DEFINE_integer('iteration', 0, "iteration at which training should start or resume")
flags.DEFINE_integer('target_update_frequency', 10000, "number of iterations after which target model is updated")


def evaluate_model(controller, agent, epsilon=FLAGS.eval_epsilon, n_episodes=FLAGS.eval_episodes):
    episode_cnt = 0
    score = 0
    evaluation_score = 0
    gif_frames = []
    controller.reset()
    no_op = 1
    for _ in range(1000000):
        if no_op > 0:
            action = 1
            no_op -= 1
        else:
            action = agent.choose_action(controller.get_state(), epsilon)
        frame, reward, is_done, life_lost = controller.step(action, evaluation=True)
        gif_frames.append(frame)
        score += reward
        if is_done:
            evaluation_score += score
            score = 0
            controller.reset()
            episode_cnt += 1
            no_op = random.randrange(FLAGS.no_op_max+1)
            if episode_cnt >= n_episodes:
                break
            gif_frames = []
        if is_done or life_lost:
            no_op = 1  # always start with no_op during evaluation
    result = evaluation_score/episode_cnt if episode_cnt > 0 else -1
    return result, gif_frames


def get_epsilon(iteration):
    epsilon = max(FLAGS.final_epsilon, FLAGS.initial_epsilon - (FLAGS.initial_epsilon - FLAGS.final_epsilon)
                / FLAGS.annealing_steps * iteration)
    return epsilon


def write_logs(model_id, iteration, seconds, score):
    file_name = os.path.join(MODEL_PATH, model_id) + '.csv'
    with open(file_name, 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([iteration, round(seconds), score])


def write_summaries(sess, loss_list, score, global_step, gif_frames):
    # write score and loss
    summ = sess.run(PERFORMANCE_SUMMARIES,
                    feed_dict={LOSS_PH: np.mean(loss_list),
                               SCORE_PH: score})
    SUMM_WRITER.add_summary(summ, global_step)

    # write gif
    # gif_frames = np.asarray(gif_frames)
    # summ_gif = sess.run(GIF_SUMMARY, feed_dict={GIF_PH: gif_frames})
    # SUMM_WRITER.add_summary(convert_tensor_to_gif_summary(summ_gif, fps=20), global_step)

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
    sess = tf.Session()

    # get start iteration (for resuming training on stored model)
    global_step = FLAGS.iteration - FLAGS.memory_start_size
    q_iteration = 0
    # start timer
    start_time = time.time()
    loss_list = []

    print("Initializing replay memory with {} states...".format(FLAGS.memory_start_size))

    # start training
    best_score = -np.inf  # keeps track of best score reached, used for saving best-so-far model
    try:
        while True:

            controller.reset()  # reset environment when done
            is_done = False

            while not is_done:
                # Choose the action -> do nothing at beginning of new episode
                action = agent.choose_action(controller.get_state(), get_epsilon(global_step))

                # interact with environment
                frame, reward, is_done, life_lost = controller.step(action)

                # add current experience to replay memory. both game over and dead are considered terminal states
                terminal = is_done or life_lost
                memory.add_experience(action, frame, reward, terminal)

                # Sample mini batch from memory and fit model
                if global_step % FLAGS.update_frequency == 0 and global_step > FLAGS.iteration:
                    batch = memory.get_minibatch()
                    hist = agent.fit_batch(batch)
                    loss_list.append(hist.history['loss'][0])
                    q_iteration += 1

                    if q_iteration == 1:
                        start_time = time.time()
                        print("Starting training...")

                if global_step % FLAGS.target_update_frequency == 0 and global_step > FLAGS.iteration:
                    agent.clone_target_model()

                # provide feedback about iteration, elapsed time, current performance
                if global_step % FLAGS.checkpoint_frequency == 0 and global_step > FLAGS.iteration:
                    score, _ = evaluate_model(evaluation_controller, agent)  # play evaluation episode to rate performance
                    cur_time = time.time()
                    m, s = divmod(cur_time-start_time, 60)
                    h, m = divmod(m, 60)
                    time_str = "%d:%02d:%02d" % (h, m, s)
                    # check if score is best so far and update model file(s)
                    is_highest = score > best_score
                    if FLAGS.use_checkpoints:
                        agent.save_checkpoint(global_step, is_highest)
                    if is_highest:
                        best_score = score
                    print("iteration {}, elapsed time: {}, score: {}, best: {}".format(global_step, time_str,
                                                                                       round(score, 2),
                                                                                       round(best_score, 2)))
                    write_logs(model_id, global_step, cur_time-start_time, score)

                    # _, gif_frames = evaluate_model(evaluation_controller, agent, epsilon=0, n_episodes=1)
                    write_summaries(sess, loss_list, score, global_step, _)
                    loss_list = []

                global_step += 1

    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    # save final state of model
    if FLAGS.use_checkpoints:
        agent.save_checkpoint(global_step)
        print("Latest checkpoint at iteration {}".format(global_step))

    env.close()
    env_test.close()


if __name__ == '__main__':
    main(sys.argv)
