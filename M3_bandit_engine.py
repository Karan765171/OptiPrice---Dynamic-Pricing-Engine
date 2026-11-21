import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, arms, unit_cost, epsilon=0.15):
        self.arms = [float(p) for p in arms]
        self.unit_cost = float(unit_cost)
        self.epsilon = float(epsilon)
        self.K = len(arms)
        self.counts = np.zeros(self.K, dtype=int)
        self.values = np.zeros(self.K, dtype=float)

    def select_arm(self, context_vector=None):
        if np.random.rand() < self.epsilon or np.min(self.counts) == 0:
            return int(np.random.randint(self.K))
        return int(np.argmax(self.values))

    def update(self, arm_idx, reward, converted=None, context_vector=None):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.values[arm_idx] += (float(reward) - self.values[arm_idx]) / n


class UCB1Bandit:
    def __init__(self, arms, unit_cost, c_factor=1.0):
        self.arms = [float(p) for p in arms]
        self.unit_cost = float(unit_cost)
        self.c_factor = float(c_factor)
        self.K = len(arms)
        self.counts = np.zeros(self.K, dtype=int)
        self.values = np.zeros(self.K, dtype=float)
        self.total_steps = 0

    def select_arm(self, context_vector=None):
        self.total_steps += 1
        for i in range(self.K):
            if self.counts[i] == 0:
                return i
        
        max_margin = np.max(np.array(self.arms) - self.unit_cost)
        ucb_scores = self.values + (self.c_factor * np.sqrt((2.0 * np.log(self.total_steps)) / self.counts) * max_margin)
        return int(np.argmax(ucb_scores))

    def update(self, arm_idx, reward, converted=None, context_vector=None):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.values[arm_idx] += (float(reward) - self.values[arm_idx]) / n


class ThompsonSamplingBandit:
    def __init__(self, arms, unit_cost):
        self.arms = [float(p) for p in arms]
        self.unit_cost = float(unit_cost)
        self.K = len(arms)
        self.alphas = np.ones(self.K, dtype=float)
        self.betas = np.ones(self.K, dtype=float)
        self.counts = np.zeros(self.K, dtype=int)
        self.values = np.zeros(self.K, dtype=float)

    def select_arm(self, context_vector=None):
        sampled_probs = np.random.beta(self.alphas, self.betas)
        expected_profits = (np.array(self.arms) - self.unit_cost) * sampled_probs
        return int(np.argmax(expected_profits))

    def update(self, arm_idx, reward, converted, context_vector=None):
        self.counts[arm_idx] += 1
        self.values[arm_idx] += (float(reward) - self.values[arm_idx]) / self.counts[arm_idx]
        if converted:
            self.alphas[arm_idx] += 1.0
        else:
            self.betas[arm_idx] += 1.0