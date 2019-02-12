import numpy as np
import random
from tensorflow import flags

ATARI_SHAPE = (105, 80, 4)
FLAGS = flags.FLAGS


class ReplayMemory:

    def __init__(self, size):

        self.size = size

        # counter and current index
        self.count = 0
        self.cur_idx = 0

        self.frames = np.empty((self.size, ATARI_SHAPE[0], ATARI_SHAPE[1]), dtype=np.uint8)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.terminal = np.empty(self.size, dtype=np.bool)

    def add_experience(self, frame, action, reward, terminal):

        self.frames[self.cur_idx, :, :] = frame
        self.actions[self.cur_idx] = action
        self.rewards[self.cur_idx] = reward
        self.terminal[self.cur_idx] = terminal

        # update counter and current index
        self.cur_idx = self.cur_idx + 1 if self.cur_idx < self.size else 0
        self.count = max(self.count, self.cur_idx)

    def get_state(self, index):
        return self.frames[index - ATARI_SHAPE[2] + 1:index + 1, :, :]

    def get_indices(self):
        indices = np.empty(FLAGS.batch_size, dtype=np.int32)
        for i in range(FLAGS.batch_size):
            while True:
                index = random.randint(FLAGS.agent_history, self.count - 1)
                # check if state crosses newest frame
                if (index >= self.cur_idx) and (index - self.cur_idx <= FLAGS.agent_history):
                    continue
                # check if state contains terminal frame
                if self.terminal[index - FLAGS.agent_history:index].any():
                    continue
                break
            indices[i] = index
        return indices

    def get_batch(self, batch_size):

        states = np.empty((batch_size, ATARI_SHAPE[2], ATARI_SHAPE[0], ATARI_SHAPE[1]), dtype=np.uint8)
        next_states = np.empty((batch_size, ATARI_SHAPE[2], ATARI_SHAPE[0], ATARI_SHAPE[1]), dtype=np.uint8)

        if self.count < ATARI_SHAPE[2]:
            raise ValueError('not enough memories to get batch')

        indices = self.get_indices()

        for i, idx in enumerate(indices):
            states[i] = self.get_state(idx - 1)
            next_states[i] = self.get_state(idx)

        return np.transpose(states, axes=(0, 2, 3, 1)), self.actions[indices], self.rewards[
            indices], np.transpose(next_states, axes=(0, 2, 3, 1)), self.terminal[indices]
