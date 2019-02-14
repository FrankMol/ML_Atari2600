import keras
import numpy as np

# defines a genetic training agent, the network is the same one as used in
# our DQN implementation (without mask) see that for comments.

class Agent:

    def __init__(self, env, model_name):
        self.number_of_actions = env.action_space.n
        self.model_name = model_name
        self.initialise_network()

    def initialise_network(self):
        ATARI_SHAPE = (105, 80, 4)
        inputs = keras.layers.Input(ATARI_SHAPE, name='frames')
        normalized = keras.layers.Lambda(lambda x: x/255.0)(inputs)
        convolutional_1 = keras.layers.Conv2D(16, (8, 8), activation="relu", strides=(4, 4), kernel_initializer=keras.initializers.glorot_normal(seed=None))(normalized)
        convolutional_2 = keras.layers.Conv2D(32, (4, 4), activation="relu", strides=(2, 2), kernel_initializer=keras.initializers.glorot_normal(seed=None))(convolutional_1)
        convolutional_flat = keras.layers.core.Flatten()(convolutional_2)
        hidden = keras.layers.Dense(256, activation="relu", kernel_initializer=keras.initializers.glorot_normal(seed=None))(convolutional_flat)
        outputs = keras.layers.Dense(self.number_of_actions, activation="softmax", kernel_initializer=keras.initializers.glorot_normal(seed=None))(hidden)
        self.model = keras.models.Model(inputs=inputs, outputs=outputs)

    def choose_action(self, state):
        action_strengths = self.model.predict(state)
        action = np.argmax(action_strengths)
        return action
