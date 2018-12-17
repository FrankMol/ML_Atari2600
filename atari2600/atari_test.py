# Import the gym module
import gym
# make env before importing tensorflow, otherwise it will not load for some reason
env_test = gym.make('BreakoutDeterministic-v4')

from atari_agent import AtariAgent
from atari_preprocessing import preprocess
from atari_controller import AtariController
import numpy as np
import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ATARI_SHAPE = (80, 80, 4)  # tensor flow backend -> channels last
# FLAGS = flags.FLAGS
MODEL_PATH = 'trained_models/'
MAX_STEPS = 1000
EVAL_EPS = 0.0


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

    # instantiate controller object
    controller = AtariController(env)
    controller.reset()

    try:
        no_op = 1
        score = 0
        total_score = 0
        n_episodes = 0
        while True:
            if no_op:
                action = 1
                no_op -= 1
            else:
                action = agent.choose_action(controller.get_state(), EVAL_EPS)

            _, reward, is_done, life_lost = controller.step(action)
            env.render()

            # update score
            score += reward

            if is_done:
                controller.reset()
                env.render()
                total_score += score
                n_episodes += 1
                print("Episode {}: Score = {}, avg score = {}".format(n_episodes, score, total_score/n_episodes))
                score = 0
                no_op = 1

            if life_lost:
                no_op = 1

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)
