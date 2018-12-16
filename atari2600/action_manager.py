import random


class ActionManager:

    def __init__(self, n_actions, no_op_max, initial_epsilon, final_epsilon, annealing_steps):
        self.n_actions = n_actions
        self.no_op_max = no_op_max
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.annealing_steps = annealing_steps

    def get_epsilon(self, iteration):
        epsilon = max(self.final_epsilon, self.initial_epsilon - (self.initial_epsilon - self.final_epsilon)
                      / self.annealing_steps * iteration)
        return epsilon

    def get_action(self, agent, state, iteration, evaluation=False):
        epsilon = 0 if evaluation else self.get_epsilon(iteration)
        action = random.randrange(self.n_actions) if random.random() < epsilon else agent.predict_action(state)
        return action
