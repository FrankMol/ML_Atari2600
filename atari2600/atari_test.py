# Import the gym module
import gym
# make env before importing tensorflow, otherwise it will not load for some reason
env = gym.make('BreakoutDeterministic-v4')

from atari_agent import AtariAgent
from atari_preprocessing import preprocess
import numpy as np
import os
import sys
import time
from tensorflow import flags

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ATARI_SHAPE = (105, 80, 4)  # tensorflow backend -> channels last
FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', 'trained_models/', "default folder for storing model files")


def update_state(state, frame):
    frame = preprocess(frame)
    frame = np.reshape([frame], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], 1))
    next_state = np.append(frame, state[:, :, :, :3], axis=3)
    return next_state


def main(argv):

    # make agent
    if len(argv) > 1 and os.path.exists(os.path.join(FLAGS.model_path, argv[1]) + '.h5'):
        agent = AtariAgent(env, argv[1])
    else:
        agent = None
        print("Give valid model name as first command line argument")
        exit(1)

    # get initial state
    frame = env.reset()
    frame = preprocess(frame)
    state = np.stack((frame, frame, frame, frame), axis=2)
    state = np.reshape([state], (1, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))

    printed_score = False
    score = 0
    steps = 0
    try:
        while True:
            frame, reward, is_done, _ = env.step(agent.choose_action(state, 0))
            env.render()
            state = update_state(state, frame)
            score += reward

            if is_done:
                env.reset()
                env.render()
                if not printed_score:
                    print("Score: {}".format(score))
                    printed_score = True

            if not printed_score:
                steps += 1

            if steps > 1000:
                print("Model could not complete single episode")
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)
