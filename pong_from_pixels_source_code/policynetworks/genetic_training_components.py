import numpy as np
import random

# this file contains functions which are used in genetic_training.py


# preprocessing by downsampeling and converting to gray scale
def preprocess(img):
    img = img[::2, ::2]
    img = np.mean(img, axis=2).astype(np.uint8)
    return img


# creates the initial state by copying the start frame 4 times
def new_environment(new_game_frame):
    game_frame = preprocess(new_game_frame)
    game_frame = np.reshape([game_frame], (1, 105, 80, 1))
    frame = np.repeat(game_frame, 4, axis=3)
    return frame


# plays a game with the agent it receives as input
def get_fitness(specimen, env, max_steps):
    frame = env.reset()
    total_reward = 0
    state = new_environment(frame)
    for t in range(max_steps):
        action = specimen.choose_action(state)  # chooses the best action according to the agent/network
        frame, reward, is_done, info = env.step(action)  # execute action
        frame = preprocess(frame)
        frame = np.reshape([frame], (1, 105, 80, 1))
        state = np.append(frame, state[:, :, :, :3], axis=3)  # create new state
        total_reward += reward
        # prevents unnecessary continuation of the game when the agent does not start (or very poorly plays) the game
        if is_done or (t == 200 and total_reward == 0):
            break
    return total_reward


# creates the new top individuals by selecting the best agents in the current generation
# if multiple agents with the same score are present priority is given to those encountered first
def create_new_top(individuals, number_of_individuals, top_individuals, number_of_top_individuals):
    new_top_indices = []
    new_top_index = 0
    for i in range(number_of_top_individuals):
        new_top_score = -1
        for j in range(number_of_individuals):
            if individuals[j][1] > new_top_score:
                new = 1
                for index in new_top_indices:
                    if index == j:
                        new = 0
                        break
                if new == 1:
                    new_top_index = j
                    new_top_score = individuals[j][1]
        new_top_indices.append(new_top_index)
        top_individuals[i][0].model.set_weights(individuals[new_top_index][0].model.get_weights())
        top_individuals[i][1] = individuals[new_top_index][1]
    return top_individuals


# checks if a new elite exists and if so returns this elite and 1
# otherwise it returns the old elite and 0
def create_new_elite(elite, top_individuals):
    is_new_elite = 0
    if elite[1] < top_individuals[0][1]:
        elite[0].model.set_weights(top_individuals[0][0].model.get_weights())
        elite[1] = top_individuals[0][1]
        is_new_elite = 1
    return elite, is_new_elite


# create the parents of the next generation, which consists of the top_individuals
# and the elite. The elite is included only once
def create_parents(elite, is_new_elite, top_individuals):
    parents = top_individuals
    if is_new_elite == 0:  # old elite, which is not part of current individuals so add to parents
        parents.append(elite)
    return parents


# creates new generation by mutating the weights of randomly selected parents
def create_new_offspring(parents, mutation_power):
    random_top = random.randint(0, len(parents) - 1)
    parent_weights = parents[random_top][0].model.get_weights()
    child_weights = []
    for layer in parent_weights:
        layer += np.random.normal(0.0, mutation_power, layer.shape)
        child_weights.append(layer)
    return child_weights


# prints info of the elite and top_individuals
def print_info(elite, top_individuals, number_of_top_individuals, generation):
    print("elite scored %d in generation %d" % (elite[1], generation))
    print("top scores in generation %d:" % generation)
    for i in range(number_of_top_individuals):
        print(top_individuals[i][1])
    print("\n")
