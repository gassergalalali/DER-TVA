"""
learning.py
Adds RL to the generator agents.
"""

import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


default_action_set = list(np.arange(-0.5, 1.6, 0.1).round(3))
default_first_action_index = default_action_set.index(0)

class Learner():
    ID = 0
    available_methods = ["roth-erev", "roth-erev modified", "basic", "multiplicative"]
    available_probability_methods = ['basic', 'gibbs-boltzmann']

    def __init__(
        self,
        action_set: list= default_action_set,
        forgetting_factor: float = 0.01,  # For Roth-Erev
        experimentation_factor: float = 0.3,  # For Roth-Erev
        learning_rate: float = 0.01, # For Multiplicative
        initial_propensity = None,
        method: str = "roth-erev",
        probability_method: str = "basic",
        first_action_index = None,
        ) -> None:
        """ General variables """
        Learner.ID += 1
        self.seed = Learner.ID
        self.random = random.Random(self.seed)
        self.action_set = list(action_set)
        self.number_of_actions = len(self.action_set)
        self.first_action_index = first_action_index if first_action_index is not None else default_first_action_index
        """ For the Roth-Erev """
        self.forgetting_factor = forgetting_factor
        self.experimentation_factor = experimentation_factor
        """ For Multiplicative """
        self.learning_rate = learning_rate
        """ Hook-up the learning method """
        self.method = method
        if method.lower() == 'roth-erev':
            self._update_propensity = self._update_propensity_roth_erev
        elif method.lower() in ['modified roth-erev', 'roth-erev modified']:
            self._update_propensity = self._update_propensity_roth_erev_modified
        elif method.lower() == 'basic':
            self._update_propensity = self._update_propensity_basic
        elif method.lower() == 'multiplicative':
            self._update_propensity = self._update_propensity_multiplicative
        else:
            raise ValueError(f"Unknown method {method}")
        # print(f"[Learner {self}] Using method {method} {self._update_propensity}")
        """ Hook-up the method to update the propensities """
        self.probability_method = probability_method
        if probability_method.lower() == 'basic':
            self._update_probability = self._update_probability_basic
        elif probability_method.lower() == 'gibbs-boltzmann':
            self._update_probability = self._update_probability_gibbs_boltzmann
        else:
            raise ValueError(f"Unknown probability method {probability_method}")


    def setup(
        self,
        number_of_iterations: int
    ):
        self.iteration_counter = 0
        self.number_of_iterations = number_of_iterations
        """ The Propensity Set """
        self.propensity = np.zeros((self.number_of_iterations + 1, self.number_of_actions))
        self.propensity[:] = np.nan
        """ The Probability Set """
        self.probability = np.zeros((self.number_of_iterations + 1, self.number_of_actions))
        self.probability[:] = np.nan
        # History of Actions and Rewards
        self.actions = np.zeros(self.number_of_iterations)
        self.actions[:] = np.nan
        self.rewards = np.zeros(self.number_of_iterations)
        self.rewards[:] = np.nan
        

    def _update_propensity_roth_erev(self):
        """ Update the propensities according to the Roth-Erev Algorithm """
        new_propensity_set = np.zeros(self.number_of_actions)
        previous_propensity_set = self.propensity[self.iteration_counter - 1]
        index_of_last_action = self.action_set.index(self.last_action)
        for i in range(self.number_of_actions):
            if i == index_of_last_action:
                new_propensity_set[i] = (
                    previous_propensity_set[i] * (1 - self.forgetting_factor)
                    + self.last_reward * (1 - self.experimentation_factor)
                )
            else:
                new_propensity_set[i] = (
                    previous_propensity_set[i] * (1 - self.forgetting_factor)
                    + (self.last_reward * self.experimentation_factor) / (self.number_of_actions - 1)
                )
        self.propensity[self.iteration_counter] = new_propensity_set

    def _update_propensity_roth_erev_modified(self):
        """ Update the propensities according to the Modified Roth-Erev Algorithm """
        new_propensity_set = np.zeros(self.number_of_actions)
        previous_propensity_set = self.propensity[self.iteration_counter - 1]
        index_of_last_action = self.action_set.index(self.last_action)
        for i in range(self.number_of_actions):
            if i == index_of_last_action:
                new_propensity_set[i] = (
                    previous_propensity_set[i] * (1 - self.forgetting_factor)
                    + self.last_reward * (1 - self.experimentation_factor)
                )
            else:
                new_propensity_set[i] = (
                    previous_propensity_set[i] * (1 - self.forgetting_factor)
                    + (previous_propensity_set[i] * self.experimentation_factor) / (self.number_of_actions - 1)
                )
        self.propensity[self.iteration_counter] = new_propensity_set

    def _update_propensity_basic(self):
        """ Update the propensities according to a basic Algorithm """
        new_propensity_set = np.zeros(self.number_of_actions)
        previous_propensity_set = self.propensity[self.iteration_counter - 1]
        index_of_last_action = self.action_set.index(self.last_action)
        for i in range(self.number_of_actions):
            if i == index_of_last_action:
                new_propensity_set[i] = previous_propensity_set[i] + self.last_reward
            else:
                new_propensity_set[i] = previous_propensity_set[i]
        self.propensity[self.iteration_counter] = new_propensity_set

    def _update_propensity_multiplicative(self):
        """ Update the propensities according to a basic Algorithm """
        new_propensity_set = np.zeros(self.number_of_actions)
        previous_propensity_set = self.propensity[self.iteration_counter - 1]
        index_of_last_action = self.action_set.index(self.last_action)
        for i in range(self.number_of_actions):
            if i == index_of_last_action:
                new_propensity_set[i] = previous_propensity_set[i] + self.learning_rate * self.last_reward
            else:
                new_propensity_set[i] = previous_propensity_set[i]
        self.propensity[self.iteration_counter] = new_propensity_set
    
    def _update_probability_gibbs_boltzmann(self):
        """ Update the probability using a Gibbs-Botzmann Distribution with a custom Cooling Factor """
        current_propensity = self.propensity[self.iteration_counter]
        new_probability_set = np.zeros(self.number_of_actions)
        boltzmann_cooling_factor = np.average(np.absolute(current_propensity))
        new_probability_set = (
            np.exp(current_propensity / boltzmann_cooling_factor)
            / np.sum(np.exp(current_propensity / boltzmann_cooling_factor))
        )
        self.probability[self.iteration_counter] = new_probability_set

    def _update_probability_basic(self):
        """ Update the probability using a Gibbs-Botzmann Distribution with a custom Cooling Factor """
        current_propensity = self.propensity[self.iteration_counter]
        new_probability_set = current_propensity / np.sum(current_propensity)
        self.probability[self.iteration_counter] = new_probability_set


    def get_action(self):
        """ choose an action"""
        if self.iteration_counter == 0:
            action = self.action_set[self.first_action_index]
        else:
            action = self.random.choices(
                population=self.action_set,
                weights=self.probability[self.iteration_counter]
                )[0]
        self.actions[self.iteration_counter] = action
        self.last_action = action
        return action
        

    def feedback_reward(
        self,
        reward,
        ):
        if self.iteration_counter == 0:
            self.propensity[0] = reward
        self.rewards[self.iteration_counter] = reward
        self.last_reward = reward
        self.iteration_counter += 1
        self._update_propensity()
        self._update_probability()
    
    def plot(self, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.subplot(421)
        plt.plot(self.actions, '.k', alpha=0.1)
        plt.ylabel("Chosen Action")
        plt.subplot(422)
        pd.Series(self.actions).value_counts().sort_index().plot.barh(ax=plt.gca())
        plt.ylabel("Density Distribution \n of Chosen Actions")
        plt.subplot(423)
        plt.plot(self.rewards, '.k')
        plt.ylabel("Reward")
        plt.subplot(413)
        sns.heatmap(np.array(self.propensity).T, ax=plt.gca(), cmap='Blues')
        plt.ylabel("Propensity Set")
        plt.gca().invert_yaxis()
        plt.subplot(414)
        sns.heatmap(np.array(self.probability).T, ax=plt.gca(), cmap='Blues')
        plt.ylabel("Probability Set")
        plt.gca().invert_yaxis()
