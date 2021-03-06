import keras
import numpy as np
import random
import os
import os.path
import json
from tensorflow import flags

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

        if os.path.exists(self.model_name + '.h5'):
            # load model and parameters
            self.model = keras.models.load_model(self.model_name + '.h5')
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

        start_states, actions, rewards, next_states, is_terminal = batch
        start_states = start_states.astype(np.float32)

        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        actions_mask = np.ones((FLAGS.batch_size, self.n_actions))
        next_q_values = self.model.predict([next_states, actions_mask])
        # next_Q_values = model.predict([next_states, np.expand_dims(np.ones(actions.shape), axis=0)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q
        q_values = rewards + FLAGS.discount_factor * np.max(next_q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        one_hot_actions = self.get_one_hot(actions)
        self.model.fit(
            [start_states, one_hot_actions], one_hot_actions * q_values[:, None],
            epochs=1, batch_size=len(start_states), verbose=0
        )

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
