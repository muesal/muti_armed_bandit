# epsilon-greedy example implementation of a multi-armed bandit
import collections
import random
import numpy as np

class Bandit:
    """
    Generic epsilon-greedy bandit that you need to improve
    """
    def __init__(self, arms, epsilon=0.1):
        """
        Initiates the bandits

        :param arms: List of arms
        :param epsilon: Epsilon value for random exploration
        """
        self.EPSILON_MIN = 0.01

        self.arms = arms
        self.epsilon = epsilon
        self.expected_values = [0] * len(arms)

        self.WINDOW = 15
        self.windows = [collections.deque(maxlen=self.WINDOW) for _ in range(len(arms))]
        self.frequencies = [0] * len(arms)
        self.sums = [0] * len(arms)

        self.iteration = 0

    def run(self):
        """
        Asks the bandit to recommend the next arm

        :return: Returns the arm the bandit recommends pulling
        """

        # Index of arm with highest expected value
        index = self.expected_values.index(max(self.expected_values))

        if min(self.frequencies) == 0:
            # index of arm which has not been pulled yet
            index = self.frequencies.index(min(self.frequencies))
        elif random.random() < self.epsilon:
            # random arm with p = eps.
            index = random.randint(0, len(self.arms) - 1)

        # decrease epsilon
        if self.epsilon > self.EPSILON_MIN:
            self.epsilon = 0.97 * self.epsilon

        return self.arms[index]

    def give_feedback(self, arm, reward):
        """
        Sets the bandit's reward for the most recent arm pull

        :param arm: The arm that was pulled to generate the reward
        :param reward: The reward that was generated
        """
        arm_index = self.arms.index(arm)
        self.sums[arm_index] = self.sums[arm_index] + reward

        # append the reward to the rewards of this arm, automatically looses oldest reward
        self.windows[arm_index].append(reward)

        reward_sum = sum(self.windows[arm_index])
        frequency = len(self.windows[arm_index])
        self.frequencies[arm_index] = frequency
        expected_value = reward_sum / frequency
        self.expected_values[arm_index] = expected_value
