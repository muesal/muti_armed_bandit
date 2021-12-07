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
        self.sums = [0] * len(arms)
        self.frequencies = [0] * len(arms)

        self.WINDOW = 38
        self.windows = np.zeros((len(arms), self.WINDOW))
        self.windows_frequencies = np.zeros((len(arms), self.WINDOW))

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

        # decrease epsilon, makes our optimization worse
        # if self.epsilon > self.EPSILON_MIN:
        #    self.epsilon = 0.97 * self.epsilon

        return self.arms[index]

    def give_feedback(self, arm, reward):
        """
        Sets the bandit's reward for the most recent arm pull

        :param arm: The arm that was pulled to generate the reward
        :param reward: The reward that was generated
        """
        # get pulled arm
        arm_index = self.arms.index(arm)

        # add the reward to the rewards of this arm
        self.sums[arm_index] = self.sums[arm_index] + reward

        # set all windows/_frequencies of this iteration to zero, set the reward and pulled arm to true
        self.windows[:, self.iteration] = 0
        self.windows[arm_index, self.iteration] = reward

        self.windows_frequencies[:, self.iteration] = 0
        self.windows_frequencies[arm_index, self.iteration] = 1

        # get the frequencies in the current window
        self.frequencies = [sum(self.windows_frequencies[row, :]) for row in range(len(self.arms))]

        # recompute all expected values
        for i in range(len(self.arms)):
            value = 0
            if self.frequencies[i] != 0:
                value = sum(self.windows[i, :]) / self.frequencies[i]
            self.expected_values[i] = value

        # increase iteration, set to zero if end of window reached
        self.iteration += 1
        if self.iteration >= self.WINDOW:
            self.iteration = 0
