# import keras
import numpy as np
import random
import os
import os.path
import json
import tensorflow as tf
from tensorflow import flags
from huber_loss import huber_loss
from DQN import DQN, TargetNetworkUpdater

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ATARI_SHAPE = (84, 84, 4)  # tensor flow backend -> channels last
FLAGS = flags.FLAGS
MODEL_PATH = 'trained_models/'

# define hyper parameters -> these can all be passed as command line arguments!
flags.DEFINE_float('discount_factor', 0.99, "discount factor used in q-learning")
flags.DEFINE_float('learning_rate', 0.00025, "learning rate used by CNN")
flags.DEFINE_float('gradient_momentum', 0.95, "gradient momentum used by CNN")
flags.DEFINE_float('sq_gradient_momentum', 0.95, "squared gradient momentum used by CNN")
flags.DEFINE_float('min_sq_gradient', 0.01, "constant added to squared gradient")


class AtariAgent:

    def __init__(self, env, model_id):
        self.n_actions = env.action_space.n
        self.model_name = os.path.join(MODEL_PATH, model_id)
        self.model = None
        self.target_model = None
        self.sess = None
        self.network_updater = None
        self.saver = None

        if os.path.exists(self.model_name + '.meta'):
            # load model and parameters
            # self.model = keras.models.load_model(self.model_name + '.h5',
            #                                      custom_objects={'huber_loss': huber_loss})
            # self.target_model = keras.models.clone_model(self.model)
            # self.clone_target_model()
            # self.load_parameters(self.model_name + '.json')
            # main DQN and target DQN networks:
            HIDDEN = 1024

            with tf.variable_scope('mainDQN'):
                self.model = DQN(self.n_actions, HIDDEN, FLAGS.learning_rate)  # (★★)
            with tf.variable_scope('targetDQN'):
                self.target_model = DQN(self.n_actions, HIDDEN)  # (★★)

            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
            TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

            self.sess = tf.Session()
            self.saver = tf.train.import_meta_graph(self.model_name + '.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_PATH))
            print("\nLoaded model '{}'".format(model_id))
        else:
            # make new model
            self.build_model()
            if FLAGS.use_checkpoints:
                self.save_checkpoint(0)
            print("\nCreated new model with name '{}'".format(model_id))

    def write_parameters(self, config_file):
        items = FLAGS.flag_values_dict()
        with open(config_file, 'w') as f:
            json.dump(items, f, indent=4)

    def load_parameters(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(FLAGS, key):
                    setattr(FLAGS, key, value)

    def write_iteration(self, config_file, iteration):
        if not os.path.exists(config_file):
            self.write_parameters(config_file)

        with open(config_file, "r") as f:
            items = json.load(f)

        items["iteration"] = iteration

        with open(config_file, "w+") as f:
            f.write(json.dumps(items, indent=4))

    def build_model(self):

        # main DQN and target DQN networks:
        HIDDEN = 1024

        with tf.variable_scope('mainDQN'):
            self.model = DQN(self.n_actions, HIDDEN, FLAGS.learning_rate)  # (★★)
        with tf.variable_scope('targetDQN'):
            self.target_model = DQN(self.n_actions, HIDDEN)  # (★★)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
        TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

        self.sess = tf.Session()
        self.sess.run(init)

        self.network_updater = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)


    def get_one_hot(self, targets):
        return np.eye(self.n_actions)[np.array(targets).reshape(-1)]

    # def fit_batch(self, batch):
    #     """Do one deep Q learning iteration.
    #
    #     Params:
    #     - model: The DQN
    #     - target_model the target DQN
    #     - gamma: Discount factor (should be 0.99)
    #     - start_states: numpy array of starting states
    #     - actions: numpy array of one-hot encoded actions corresponding to the start states
    #     - rewards: numpy array of rewards corresponding to the start states and actions
    #     - next_states: numpy array of the resulting states corresponding to the start states and actions
    #     - is_terminal: numpy boolean array of whether the resulting state is terminal
    #
    #     """
    #
    #     start_states, actions, rewards, next_states, is_terminal = batch
    #     start_states = start_states.astype(np.float32)
    #
    #     # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    #     actions_mask = np.ones((FLAGS.batch_size, self.n_actions))
    #     next_q_values = self.target_model.predict([next_states, actions_mask])
    #     # next_Q_values = model.predict([next_states, np.expand_dims(np.ones(actions.shape), axis=0)])
    #     # The Q values of the terminal states is 0 by definition, so override them
    #     next_q_values[is_terminal] = 0
    #     # The Q values of each start state is the reward + gamma * the max next state Q
    #     q_values = rewards + FLAGS.discount_factor * np.max(next_q_values, axis=1)
    #     # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    #     # the targets by the actions.
    #     one_hot_actions = self.get_one_hot(actions)
    #     self.model.fit(
    #         [start_states, one_hot_actions], one_hot_actions * q_values[:, None],
    #         epochs=1, batch_size=len(start_states), verbose=0
    #     )

    def fit_batch(self, batch):
        """
        Args:
            batch
        Returns:
            loss: The loss of the minibatch, for tensorboard
        Draws a minibatch from the replay memory, calculates the
        target Q-value that the prediction Q-value is regressed to.
        Then a parameter update is performed on the main DQN.
        """
        # Draw a minibatch from the replay memory
        start_states, actions, rewards, next_states, is_terminal = batch
        # The main network estimates which action is best (in the next
        # state s', new_states is passed!)
        # for every transition in the minibatch
        arg_q_max = self.sess.run(self.model.best_action, feed_dict={self.model.input: next_states})
        # The target network estimates the Q-values (in the next state s', new_states is passed!)
        # for every transition in the minibatch
        q_vals = self.sess.run(self.target_model.q_values, feed_dict={self.target_model.input: next_states})
        double_q = q_vals[range(FLAGS.batch_size), arg_q_max]
        # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
        # if the game is over, targetQ=rewards
        target_q = rewards + (FLAGS.discount_factor * double_q * (1 - is_terminal))
        # Gradient descend step to update the parameters of the main network
        loss, _ = self.sess.run([self.model.loss, self.model.update],
                              feed_dict={self.model.input: start_states,
                                         self.model.target_q: target_q,
                                         self.model.action: actions})
        return loss

    # def choose_action(self, state, epsilon):
    #     state = state.astype(np.float32)
    #     if random.random() < epsilon:
    #         action = random.randrange(self.n_actions)
    #     else:
    #         mask = np.ones(self.n_actions).reshape(1, self.n_actions)
    #         q_values = self.model.predict([state, mask])
    #         action = np.argmax(q_values)
    #     return action

    def choose_action(self, state, epsilon):
        state = state.astype(np.float32)
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            action = self.sess.run(self.model.best_action, feed_dict={self.model.input: state})[0]
        return action

    # def save_checkpoint(self, iteration, new_best=False):
    #     self.model.save(self.model_name + '.h5')
    #     self.write_iteration(self.model_name + '.json', iteration)
    #     if new_best:
    #         self.model.save(self.model_name + '_best.h5')
    #         self.write_iteration(self.model_name + '_best.json', iteration)

    def save_checkpoint(self, iteration, new_best=False):
        self.saver.save(self.sess, self.model_name, global_step=iteration)
        self.write_iteration(self.model_name + '.json', iteration)
        if new_best:
            self.saver.save(self.sess, self.model_name, global_step=iteration)
            self.write_iteration(self.model_name + '_best.json', iteration)




    # def clone_target_model(self):
    #     self.target_model.set_weights(self.model.get_weights())

    def clone_target_model(self):
        self.network_updater.update_networks(self.sess)
