import keras
import numpy as np
import random
import os
import os.path
import json
from tensorflow import flags
from huber_loss import huber_loss
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
flags.DEFINE_string('loss', 'huber', 'loss function used by optimizer')

def mean(x):
    return K.mean(x, axis=1, keepdims=True)

class AtariAgent:

    def __init__(self, env, model_id):
        self.n_actions = env.action_space.n
        self.model_name = os.path.join(MODEL_PATH, model_id)
        self.model = None
        self.target_model = None

        if os.path.exists(self.model_name + '.h5'):
            # load model and parameters
            self.model = keras.models.load_model(self.model_name + '.h5',
                                                 custom_objects={'huber_loss': huber_loss})
            if FLAGS.use_target_model:
                self.target_model = keras.models.clone_model(self.model)
                self.clone_target_model()
            self.load_parameters(self.model_name + '.json')
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
        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        actions_input = keras.layers.Input((self.n_actions,), name='mask')

        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
        conv_1 = keras.layers.Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)
        conv_2 = keras.layers.Conv2D(32, (4, 4), activation="relu", strides=(2, 2))(conv_1)
        conv_flattened = keras.layers.core.Flatten()(conv_2)

        # value_stream = keras.layers.core.Flatten()(conv_2)
        # advantage_stream = keras.layers.core.Flatten()(conv_2)

        value_hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        advantage_hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)

        advantage_output = keras.layers.Dense(self.n_actions)(advantage_hidden)
        value_output = keras.layers.Dense(1)(value_hidden)

        tmp = keras.layers.Lambda(mean)(advantage_output)
        tmp2 = keras.layers.merge.Subtract()([advantage_output, tmp])
        output = keras.layers.merge.Add()([value_output, tmp2])

        filtered_output = keras.layers.merge.Multiply()([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        optimizer = keras.optimizers.RMSprop(lr=FLAGS.learning_rate,
                                             rho=FLAGS.gradient_momentum,
                                             epsilon=FLAGS.min_sq_gradient)
        # check loss function and compile model
        if FLAGS.loss == 'huber':
            self.model.compile(optimizer, loss=huber_loss)
        elif FLAGS.loss == 'mse':
            print('using mse loss')
            self.model.compile(optimizer, loss='mse')

        # set up target model
        if FLAGS.use_target_model:
            self.target_model = keras.models.clone_model(self.model)

    # def build_model(self):
    #
    #     frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    #     actions_input = keras.layers.Input((self.n_actions,), name='mask')
    #
    #     normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    #     conv_1 = keras.layers.Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)
    #     conv_2 = keras.layers.Conv2D(32, (4, 4), activation="relu", strides=(2, 2))(conv_1)
    #     conv_flattened = keras.layers.core.Flatten()(conv_2)
    #     hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    #     output = keras.layers.Dense(self.n_actions)(hidden)
    #     filtered_output = keras.layers.merge.Multiply()([output, actions_input])
    #
    #     self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    #
    #     optimizer = keras.optimizers.RMSprop(lr=FLAGS.learning_rate,
    #                                          rho=FLAGS.gradient_momentum,
    #                                          epsilon=FLAGS.min_sq_gradient)
    #     # check loss function and compile model
    #     if FLAGS.loss == 'huber':
    #         self.model.compile(optimizer, loss=huber_loss)
    #     elif FLAGS.loss == 'mse':
    #         print('using mse loss')
    #         self.model.compile(optimizer, loss='mse')
    #
    #     # set up target model
    #     if FLAGS.use_target_model:
    #         self.target_model = keras.models.clone_model(self.model)

    def get_one_hot(self, targets):
        return np.eye(self.n_actions)[np.array(targets).reshape(-1)]

    def fit_batch(self, batch):
        start_states, actions, rewards, next_states, is_terminal = batch
        start_states = start_states.astype(np.float32)

        actions_mask = np.ones((FLAGS.batch_size, self.n_actions))

        if FLAGS.use_target_model:
            next_q_values = self.target_model.predict([next_states, actions_mask])
        else:
            next_q_values = self.model.predict([next_states, actions_mask])

        next_q_values[is_terminal] = 0

        q_values = rewards + FLAGS.discount_factor * np.max(next_q_values, axis=1)

        one_hot_actions = self.get_one_hot(actions)
        loss = self.model.fit(
            [start_states, one_hot_actions], one_hot_actions * q_values[:, None],
            epochs=1, batch_size=len(start_states), verbose=0
        )
        return loss

    def choose_action(self, state, epsilon):
        state = state.astype(np.float32)
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            mask = np.ones(self.n_actions).reshape(1, self.n_actions)
            q_values = self.model.predict([state, mask])
            action = np.argmax(q_values)
        return action

    def save_checkpoint(self, iteration, new_best=False):
        self.model.save(self.model_name + '.h5')
        self.write_iteration(self.model_name + '.json', iteration)
        if new_best:
            self.model.save(self.model_name + '_best.h5')
            self.write_iteration(self.model_name + '_best.json', iteration)

    def clone_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
