import gym
from genetic_agent import Agent
import genetic_training_components

# initializes the breakout environment, can in principle be set to any game in the gym ALE environment
env = gym.make('BreakoutDeterministic-v4')
env.reset()
# hardcoded parameters of the genetic algorithm, note there is always only 1 elite
number_of_top_individuals = 10
number_of_individuals = 50
number_of_generations = 100
max_steps = 10000 # maximum number of game steps allowed, if this es exceeded the game will be stopped
mutation_power = 0.02

# the following set the lists containing agent, which are randomly initialised,
# and score pairs for elite, top_individuals and individuals
# note that it is not really necessary to randomly initialise elite and top_individuals
# since they will not be used in the first generation
elite_agent = Agent(env, "elite_agent")
elite = [elite_agent, -1]
top_individuals = []
for i in range(number_of_top_individuals):
    top_agent = Agent(env, "top_agent%d" % i)
    top_individuals.append([top_agent, -1])
individuals = []
for i in range(number_of_individuals):
    agent = Agent(env, "agent%d" % i)
    individuals.append([agent, -1])

for generation in range(number_of_generations):
    # evaluate the fitness of all individuals in this generation by playing the game
    for i in range(number_of_individuals):
        individuals[i][1] = genetic_training_components.get_fitness(individuals[i][0], env, max_steps)
    # select the individuals in the current generation with the highest score and put them in top_individuals
    top_individuals = genetic_training_components.create_new_top(individuals, number_of_individuals, top_individuals, number_of_top_individuals)
    # receives the new elite and 1 in is_new_elite if the elite is different from the last generation
    # and 0 otherwise
    elite, is_new_elite = genetic_training_components.create_new_elite(elite, top_individuals)
    # print information about the elite and top_individuals of the current generation
    genetic_training_components.print_info(elite, top_individuals, number_of_top_individuals, generation)
    # create a list of parents from which the next generation will be created
    parents = genetic_training_components.create_parents(elite, is_new_elite, top_individuals)
    # create a new generation by mutating parents
    for i in range(number_of_individuals):
        individuals[i][0].model.set_weights(genetic_training_components.create_new_offspring(parents, mutation_power))
