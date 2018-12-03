import keras
import numpy as np
import time
import random
import gym
from atari_preprocessing import preprocess
import numpy as np
import os
import os.path
from collections import deque

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# We assume a theano backend here, so the "channels" are first.
# ATARI_SHAPE = (4, 105, 80)
ATARI_SHAPE = (105, 80, 4)
BATCH_SIZE = 32
ACTIONS_SIZE = 3
MAX_ITERATIONS = 10000000
OBSERVE_ITERATIONS = 50000
START_ITERATION = 0 #889000

class AtariAgent:

    def __init__(self, env, model_name):
        self.model_name = model_name
        model_path = "trained_models/"+model_name+".h5"
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print("loaded model")
        else:
            self.atari_model(ACTIONS_SIZE)

    def atari_model(self, n_actions):
        # from keras import backend as K
        # K.set_image_dim_ordering('th')

        # With the functional API we need to define the inputs.
        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        actions_input = keras.layers.Input((n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        # conv_1 = keras.layers.convolutional.Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)
        conv_1 = keras.layers.Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)

        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        # conv_2 = keras.layers.convolutional.Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)
        conv_2 = keras.layers.Conv2D(32, (4, 4), activation="relu", strides=(2, 2))(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(n_actions)(hidden)
        # Finally, we multiply the output by the mask!
        # filtered_output = keras.layers.merge([output, actions_input], mode='mul')
        filtered_output = keras.layers.merge.Multiply()([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)

        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer, loss='mse')

    def get_one_hot(self, targets, nb_classes=ACTIONS_SIZE):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]

    def fit_batch(self, model, gamma, start_states, actions, rewards, next_states, is_terminal):
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
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        actions_mask = np.ones((BATCH_SIZE, ACTIONS_SIZE))
        next_Q_values = model.predict([next_states, actions_mask])
        # next_Q_values = model.predict([next_states, np.expand_dims(np.ones(actions.shape), axis=0)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q
        Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        one_hot_actions = self.get_one_hot(actions)
        # print(one_hot_actions)
        model.fit(
            [start_states, one_hot_actions], one_hot_actions * Q_values[:, None],
            nb_epoch=1, batch_size=len(start_states), verbose=0
        )

    def choose_best_action(self, model, state, actions):
        # state = np.expand_dims(state, axis=0)
        mask = np.ones(ACTIONS_SIZE).reshape(1, ACTIONS_SIZE)
        # q_values = model.predict([state, np.ones(ACTIONS_SIZE)])
        q_values = model.predict([state, mask])
        return np.argmax(q_values)

    def get_epsilon_for_iteration(self, iteration):
        epsilon = max(0.1, 1-(1-0.1)/1000000*iteration)
        return epsilon

    # def q_iteration(self, env, state, iteration, memory): # model was second parameter (after env)
    #     # Choose epsilon based on the iteration
    #     epsilon = self.get_epsilon_for_iteration(iteration)
    #
    #     # Choose the action
    #     if random.random() < epsilon:
    #         action = random.randrange(ACTIONS_SIZE)
    #     else:
    #         action = self.choose_best_action(self.model, state, env.action_space)
    #
    #     # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    #     new_state = np.zeros(ATARI_SHAPE)
    #     new_reward = 0
    #     new_is_done = False
    #     for i in range(4):
    #         new_frame, reward, is_done, _ = env.step(action+1)
    #         new_state[i] = preprocess(new_frame)
    #         new_reward = max(new_reward, reward)
    #         new_is_done = new_is_done or is_done
    #     memory.append((state, action, new_reward, new_state, new_is_done))
    #
    #     if new_is_done:
    #         env.reset()
    #
    #     # Sample and fit
    #     batch = random.sample(memory, BATCH_SIZE)
    #     # unpack batch
    #     start_states = np.zeros((32, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    #     next_states = np.zeros((32, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    #     actions, rewards, is_terminal = [], [], []
    #
    #     for idx, val in enumerate(batch):
    #         start_states[idx] = val[0]
    #         next_states[idx] = val[3]
    #         actions.append(val[1])
    #         rewards.append(val[2])
    #         is_terminal.append(val[4])
    #     self.fit_batch(self.model, 0.99, start_states, actions, rewards, next_states, is_terminal)
    #     return new_state

    def train(self, env):
        # reset environment
        env.reset()
        # get initial state
        frame = env.reset()
        frame = preprocess(frame)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))

        # initialize memory
        memory = deque(maxlen=1000000)

        # first fill memory
        print("Initializing memory with {} states...".format(OBSERVE_ITERATIONS))
        for iteration in range(OBSERVE_ITERATIONS):
            # Choose epsilon based on the iteration
            epsilon = self.get_epsilon_for_iteration(START_ITERATION)

            # Choose the action
            if random.random() < epsilon:
                action = random.randrange(ACTIONS_SIZE)
            else:
                action = self.choose_best_action(self.model, state, env.action_space)

            real_action=action+1
            frame, reward, is_done, _ = env.step(real_action)
            frame = preprocess(frame)
            frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
            next_state = np.append(frame, state[:, :, :, :3], axis=3)
            memory.append((state, action, reward, next_state, is_done))

            if is_done:
                env.reset()
            state = next_state
        print("Memory initialized, starting training")

        score = 0
        max_score = 0
        # start training
        for iteration in range(START_ITERATION, MAX_ITERATIONS):

            # Choose epsilon based on the iteration
            epsilon = self.get_epsilon_for_iteration(iteration)

            # Choose the action
            if random.random() < epsilon:
                action = random.randrange(ACTIONS_SIZE) # add 1 because this because both 0 and 1 -> do nothing
            else:
                action = self.choose_best_action(self.model, state, env.action_space)
            real_action = action + 1

            # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
            frame, reward, is_done, _ = env.step(real_action)
            frame = preprocess(frame)
            frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
            next_state = np.append(frame, state[:, :, :, :3], axis=3)

            memory.append((state, action, reward, next_state, is_done))

            score += reward
            if is_done:
                env.reset()
                max_score = max(max_score, score)
                score = 0

            # Sample and fit
            batch = random.sample(memory, BATCH_SIZE)
            # unpack batch
            start_states = np.zeros((32, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
            next_states = np.zeros((32, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
            actions, rewards, is_terminal = [], [], []

            for idx, val in enumerate(batch):
                start_states[idx] = val[0]
                next_states[idx] = val[3]
                actions.append(val[1])
                rewards.append(val[2])
                is_terminal.append(val[4])
            self.fit_batch(self.model, 0.99, start_states, actions, rewards, next_states, is_terminal)

            if iteration % 500 ==0:
                print("iteration: {}, max score = {}".format(iteration,max_score))
                max_score = 0;

            if iteration % 1000 == 0:
                self.model.save("trained_models/{}.h5".format(self.model_name))

            state = next_state
        env.close()

    def test(self, env):
        frame = env.reset()
        env.step(1)
        frame = preprocess(frame)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
        env.render()
        start_life=5
        while True:
            action = self.choose_best_action(self.model, state, env.action_space)
            frame, reward, is_done, info = env.step(action+1)
            # if start_life > info['ale.lives']:
            #     start_life = info['ale.lives']
            #     frame, reward, is_done, info = env.step(1)
            env.render()
            time.sleep(0.1)
            frame = preprocess(frame)
            frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
            state = np.append(frame, state[:, :, :, :3], axis=3)
            if is_done:
                start_life = 5
                env.reset()
        env.close()