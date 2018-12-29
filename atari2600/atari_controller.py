import numpy as np
from atari_preprocessing import preprocess

ATARI_SHAPE = (105, 80, 4)


class AtariController:

    def __init__(self, env, agent_history_length=4):
        self.env = env
        self.lives = 0  # will be initialized to max lives in first step
        self.state = None
        self.agent_history_length = agent_history_length

    def reset(self):
        frame = self.env.reset()
        frame = preprocess(frame)
        self.lives = 0
        frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
        self.state = np.repeat(frame, self.agent_history_length, axis=3)
        return

    def step(self, action, evaluation=False):
        raw_frame, reward, is_done, info = self.env.step(action)

        # check if life lost
        life_lost = True if info['ale.lives'] < self.lives else False
        self.lives = info['ale.lives']

        # preprocess frame
        frame = preprocess(raw_frame)
        state_frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
        self.state = np.append(self.state[:, :, :, 1:], state_frame, axis=3)
        # clip reward
        reward = np.sign(reward)

        # return raw frame when evaluating, for making gif
        if evaluation:
            return raw_frame, reward, is_done, life_lost
        else:
            return frame, reward, is_done, life_lost

    def get_state(self):
        return self.state
