import numpy as np
from atari_preprocessing import preprocess

ATARI_SHAPE = (80, 80, 4)

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
        self.state = np.repeat(frame, self.agent_history_length, axis=2)
        return frame

    def step(self, action):
        frame, reward, is_done, info = self.env.step(action)

        # check if life lost
        if info['ale.lives'] < self.lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = is_done
        self.lives = info['ale.lives']

        # preprocess frame
        frame = preprocess(frame)
        frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
        self.state = np.append(frame, self.state[:, :, :, :3], axis=3)

        # clip reward
        reward = np.sign(reward)

        return frame, reward, is_done, terminal_life_lost
