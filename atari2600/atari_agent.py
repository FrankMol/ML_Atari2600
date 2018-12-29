import keras
import numpy as np
import random
import os
import os.path
import json
from tensorflow import flags
from huber_loss import huber_loss
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop, Adam
from keras import backend as K

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ATARI_SHAPE = (105, 80, 4)  # tensor flow backend -> channels last
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

        self.q_model = self.build_q_model()
        self.v_model = self.build_v_model()
        self.target_v_model = self.build_v_model()

        self.optimizer = self.optimizer()

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

    def build_q_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu', input_shape=ATARI_SHAPE))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.n_actions))
        model.compile(loss='mse', optimizer=RMSprop(lr=FLAGS.learning_rate, rho=0.95, epsilon=0.01))

        return model

    def build_v_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu', input_shape=ATARI_SHAPE))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=RMSprop(
            lr=FLAGS.learning_rate, rho=0.95, epsilon=0.01))

        return model

    def optimizer(self):

        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.q_model.output

        a_one_hot = K.one_hot(a, self.n_actions)
        q_value = K.sum(py_x * a_one_hot, axis=1)

        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part

        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(
            self.q_model.trainable_weights, [], loss)
        train = K.function([self.q_model.input, a, y], [loss], updates=updates)

        return train

    def get_one_hot(self, targets):
        return np.eye(self.n_actions)[np.array(targets).reshape(-1)]

    def fit_batch(self, batch):
        """Do one deep Q learning iteration.

        Params:
        - model: The DQN
        - target_model the target DQN
        - gamma: Discount factor (should be 0.99)
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal

        """

        start_states, actions, rewards, next_states, is_terminal = batch
        start_states = start_states.astype(np.float32)

        # # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        # actions_mask = np.ones((FLAGS.batch_size, self.n_actions))
        # next_q_values = self.target_model.predict([next_states, actions_mask])
        # # next_Q_values = model.predict([next_states, np.expand_dims(np.ones(actions.shape), axis=0)])
        # # The Q values of the terminal states is 0 by definition, so override them
        # next_q_values[is_terminal] = 0
        # # The Q values of each start state is the reward + gamma * the max next state Q
        # q_values = rewards + FLAGS.discount_factor * np.max(next_q_values, axis=1)
        # # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # # the targets by the actions.
        # one_hot_actions = self.get_one_hot(actions)
        # self.v_model.fit(
        #     [start_states, one_hot_actions], one_hot_actions * q_values[:, None],
        #     epochs=1, batch_size=len(start_states), verbose=0
        # )
        #
        #
        #

        v_target = np.zeros((len(start_states),))

        q_target = self.q_model.predict(start_states)

        v_target_value = self.target_v_model.predict(next_states)

        q_targets = list()

        for i in range(len(start_states)):

            if is_terminal[i]:
                v_target[i] = rewards[i]
                q_target[i][actions[i]] = rewards[i]

            else:
                v_target[i] = rewards[i] + \
                              FLAGS.discount_factor * v_target_value[i]
                q_target[i][actions[i]] = rewards[i] + \
                                         FLAGS.discount_factor * v_target_value[i]

            q_targets.append(q_target[i][actions[i]])

        loss = self.optimizer([start_states, actions, q_targets])

        self.v_model.fit(start_states, v_target, epochs=1, verbose=0)

    def choose_action(self, state, epsilon):
        state = state.astype(np.float32)
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            mask = np.ones(self.n_actions).reshape(1, self.n_actions)
            q_values = self.v_model.predict(state)
            action = np.argmax(q_values)
        return action

    def save_checkpoint(self, iteration, new_best=False):
        self.v_model.save(self.model_name + '.h5')
        self.write_iteration(self.model_name + '.json', iteration)
        if new_best:
            self.v_model.save(self.model_name + '_best.h5')
            self.write_iteration(self.model_name + '_best.json', iteration)

    def clone_target_model(self):
        self.target_v_model.set_weights(self.v_model.get_weights())
