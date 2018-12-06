import keras
import numpy as np
import time
import random
from atari_preprocessing import preprocess
import os
import os.path
import json
from tensorflow import flags
from collections import deque
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ATARI_SHAPE = (105, 80, 4) # tensorflow backend -> channels last
FLAGS = flags.FLAGS
MODELS_PATH = 'trained_models/'

new_model_name = "new_model_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + ".h5"

# define hyperparameters -> these can all be passed ass command line arguments!
flags.DEFINE_integer('batch_size', 32, "mini batch size")
flags.DEFINE_integer('memory_size', 1000000, "max number of stored states from which batch is sampled")
flags.DEFINE_integer('memory_start_size', 50000, "number of states with which the memory is initialized")
flags.DEFINE_integer('agent_history', 4, "number of frames in each state")
flags.DEFINE_float('discount_factor', 0.99, "discount factor used in q-learning")
flags.DEFINE_integer('update_frequency', 4, "number of actions played by agent between each q-iteration")
flags.DEFINE_float('learning_rate', 0.00025, "learning rate used by CNN")
flags.DEFINE_float('gradient_momentum', 0.95, "gradient momentum used by CNN")
flags.DEFINE_float('sq_gradient_momentum', 0.95, "squared gradient momentum used by CNN")
flags.DEFINE_float('min_sq_gradient', 0.01, "constant added to squared gradient")
flags.DEFINE_float('initial_epsilon', 1, "initial value of epsilon used for exploration of state space")
flags.DEFINE_float('final_epsilon', 0.1, "final value of epsilon used for exploration of state space")
flags.DEFINE_integer('final_exploration_frame', 1000000,
                   "frame at which final exploration reached") # LET OP: frame of q-iteration?
flags.DEFINE_integer('no_op_max', 30, "max number of do nothing actions at beginning of episode")
flags.DEFINE_string('model_name', new_model_name, "file name to which model will be saved")
flags.DEFINE_string('load_model', '', "load model specificed by string")

class AtariAgent:

    def __init__(self, env):
        self.n_actions = env.action_space.n
        if FLAGS.load_model == '':
            self.model_name = FLAGS.model_name
            self.build_model()
            print("Created new model '{}'".format(FLAGS.model_name))
        else:
            if os.path.exists(FLAGS.model_name):
                self.model = keras.models.load_model(FLAGS.model_name)
                self.model_name = FLAGS.model_name
                print("Loaded model "+FLAGS.model_name)
            else:
                print("Model with name {} could not be found".format(FLAGS.model_name))
                exit(1)

    def load_parameters(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            for name, value in config.items():
                FLAGS.__flags[name].value = value

    def build_model(self):
        # from keras import backend as K
        # K.set_image_dim_ordering('th')

        # With the functional API we need to define the inputs.
        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        actions_input = keras.layers.Input((self.n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        conv_1 = keras.layers.Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)

        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = keras.layers.Conv2D(32, (4, 4), activation="relu", strides=(2, 2))(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(self.n_actions)(hidden)
        # Finally, we multiply the output by the mask!
        filtered_output = keras.layers.merge.Multiply()([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        optimizer = keras.optimizers.RMSprop(lr=FLAGS.learning_rate,
                                             rho=FLAGS.gradient_momentum,
                                             epsilon=FLAGS.min_sq_gradient)
        self.model.compile(optimizer, loss='mse')

    def get_one_hot(self, targets):
        return np.eye(self.n_actions)[np.array(targets).reshape(-1)]

    def fit_batch(self, batch):
        """Do one deep Q learning iteration.

        Params:
        - model: The DQN
        - gamma: Discount factor (should be 0.99)
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal

        """
        # first decode batch into arrays of states, rewards and actions
        start_states = np.zeros((32, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
        next_states = np.zeros((32, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
        actions, rewards, is_terminal = [], [], []

        for idx, val in enumerate(batch):
            start_states[idx] = val[0]
            actions.append(val[1])
            rewards.append(val[2])
            next_states[idx] = val[3]
            is_terminal.append(val[4])

        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        actions_mask = np.ones((FLAGS.batch_size, self.n_actions))
        next_Q_values = self.model.predict([next_states, actions_mask])
        # next_Q_values = model.predict([next_states, np.expand_dims(np.ones(actions.shape), axis=0)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q
        Q_values = rewards + FLAGS.discount_factor * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        one_hot_actions = self.get_one_hot(actions)
        self.model.fit(
            [start_states, one_hot_actions], one_hot_actions * Q_values[:, None],
            nb_epoch=1, batch_size=len(start_states), verbose=0
        )

    # def choose_best_action(self, model, state):
    #     mask = np.ones(self.n_actions).reshape(1, self.n_actions)
    #     q_values = model.predict([state, mask])
    #     return np.argmax(q_values)

    def choose_action(self, state, epsilon):

        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            mask = np.ones(self.n_actions).reshape(1, self.n_actions)
            q_values = self.model.predict([state, mask])
            action = np.argmax(q_values)
        return action

    def save_model_to_file(self):
        self.model.save(os.path.join(MODELS_PATH,self.model_name))

    # def get_epsilon_for_iteration(self, iteration):
    #     epsilon = max(FLAGS.final_epsilon, FLAGS.initial_epsilon - (FLAGS.initial_epsilon - FLAGS.final_epsilon)
    #                   / FLAGS.final_exploration_frame*iteration)
    #     return epsilon

    # def train(self, env):
    #     # reset environment
    #     env.reset()
    #     # get initial state
    #     frame = env.reset()
    #     frame = preprocess(frame)
    #     state = np.stack((frame, frame, frame, frame), axis=2)
    #     state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    #
    #     # initialize memory
    #     memory = deque(maxlen=FLAGS.memory_size)
    #
    #     # first fill memory
    #     print("Initializing memory with {} states...".format(FLAGS.memory_start_size))
    #     for iteration in range(FLAGS.memory_start_size):
    #         # Choose epsilon based on the iteration
    #         epsilon = self.get_epsilon_for_iteration(START_ITERATION)
    #
    #         # Choose the action
    #         if random.random() < epsilon:
    #             action = env.action_space.sample()
    #         else:
    #             action = self.choose_best_action(self.model, state)
    #
    #         real_action=action+1
    #         frame, reward, is_done, _ = env.step(real_action)
    #         frame = preprocess(frame)
    #         frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    #         next_state = np.append(frame, state[:, :, :, :3], axis=3)
    #         memory.append((state, action, reward, next_state, is_done))
    #
    #         if is_done:
    #             env.reset()
    #         state = next_state
    #     print("Memory initialized, starting training")
    #
    #     score = 0
    #     max_score = 0
    #     # start training
    #     for iteration in range(START_ITERATION, MAX_ITERATIONS):
    #
    #         # Choose epsilon based on the iteration
    #         epsilon = self.get_epsilon_for_iteration(iteration)
    #
    #         # Choose the action
    #         if random.random() < epsilon:
    #             action = random.randrange(ACTIONS_SIZE) # add 1 because this because both 0 and 1 -> do nothing
    #         else:
    #             action = self.choose_best_action(self.model, state, env.action_space)
    #         real_action = action + 1
    #
    #         # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    #         frame, reward, is_done, _ = env.step(real_action)
    #         frame = preprocess(frame)
    #         frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    #         next_state = np.append(frame, state[:, :, :, :3], axis=3)
    #
    #         memory.append((state, action, reward, next_state, is_done))
    #
    #         score += reward
    #         if is_done:
    #             env.reset()
    #             max_score = max(max_score, score)
    #             score = 0
    #
    #         # Sample and fit
    #         batch = random.sample(memory, FLAGS.batch_size)
    #         # unpack batch
    #
    #         self.fit_batch(batch)
    #
    #         if iteration % 500 ==0:
    #             print("iteration: {}, max score = {}".format(iteration,max_score))
    #             max_score = 0
    #
    #         if iteration % 1000 == 0:
    #             self.model.save("trained_models/{}.h5".format(self.model_name))
    #
    #         state = next_state
    #     env.close()

    def test(self, env):
        frame = env.reset()
        env.step(1)
        frame = preprocess(frame)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
        env.render()
        while True:
            action = self.choose_best_action(self.model, state)
            frame, reward, is_done, info = env.step(action+1)
            env.render()
            time.sleep(0.1)
            frame = preprocess(frame)
            frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
            state = np.append(frame, state[:, :, :, :3], axis=3)
            if is_done:
                env.reset()
        env.close()
