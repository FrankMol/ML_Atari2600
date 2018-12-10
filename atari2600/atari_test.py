# Import the gym module
import gym
# make env before importing tensorflow, otherwise it will not load for some reason
env_test = gym.make('BreakoutDeterministic-v4')

from atari_agent import AtariAgent
from atari_preprocessing import preprocess
import numpy as np
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ATARI_SHAPE = (105, 80, 4)  # tensor flow backend -> channels last
# FLAGS = flags.FLAGS
MODEL_PATH = 'trained_models/'
MAX_STEPS = 1000


def update_state(state, frame):
    frame = preprocess(frame)
    frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    next_state = np.append(frame, state[:, :, :, :3], axis=3)
    return next_state


# reset environment and get first state
def get_start_state(env):
    frame = env.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    return state


def play_episode(env, agent):
    score = 0
    is_done = 0
    steps = 0
    state = get_start_state(env)
    while not is_done:
        frame, reward, is_done, _ = env.step(agent.choose_action(state, 0))
        state = update_state(state, frame)
        score += reward
        steps += 1
        # no points in large no of steps means agent only plays no-op. Stop testing
        if (steps > 1000 and score < 30) or steps > 20000:
            break
    if is_done:
        return score
    else:
        return -1


def main(argv):

    env = env_test
    # check if model exists and load model
    if len(argv) > 1:
        agent = AtariAgent(env, argv[1])
    else:
        agent = None
        print("Give valid model name as first command line argument")
        exit(1)

    # get initial state
    state = get_start_state(env)

    # play single episode to get score
    score = play_episode(env, agent)
    if score == -1:
        print("Agent did not finish episode within {} steps".format(MAX_STEPS))
        return
    else:
        print("Agent performance: {}".format(int(score)))

    try:
        while True:
            frame, reward, is_done, _ = env.step(agent.choose_action(state, 0))
            env.render()
            state = update_state(state, frame)

            if is_done:
                env.reset()
                env.render()

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)
