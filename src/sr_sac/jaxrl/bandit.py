from typing import List
import numpy as np


class ExpWeights(object):
    """
    pojedynczy bandyta z papieru tactical optimizm and pesimizm
    """

    def __init__(
        self,
        arms: List = [0, 1],
        lr: float = 0.1,
        window: int = 5,
        decay: float = 0.9,
        init: float = 0.0,
        use_std: bool = True,
    ) -> None:
        """Initialize bandit.
        Args:
            arms (List, optional): Arm values. Defaults to [-1, 0, 1].
            lr (float, optional): Learning rate. Defaults to 0.2.
            window (int, optional): Window to normalize over. Defaults to 5.
            decay (float, optional): Decay rate for probability. Defaults to 0.9.
            init (float, optional): Weight initialization. Defaults to 0.0.
            use_std (bool, optional): Use std to normalize feedback. Defaults to True.
        """

        self.arms = arms
        self.w = {i: init for i in range(len(self.arms))}
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.decay = decay
        self.use_std = use_std

        self.choices = [self.arm]
        self.data = []

    def sample(self) -> float:
        """Sample from distribution.
        Returns:
            float: The value of the sampled arm.
        """
        p = [np.exp(x) for x in self.w.values()]
        p /= np.sum(p)  # normalize to make it a distribution
        self.arm = np.random.choice(range(0, len(p)), p=p)

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)

        return self.value

    def get_probs(self) -> List:
        """Get arm probabilities.
        Returns:
            List: probabilities for each arm.
        """
        p = [np.exp(x) for x in self.w.values()]
        p /= np.sum(p)  # normalize to make it a distribution
        return p

    def update_dists(self, feedback: float, norm: float = 1.0) -> None:
        """Update distribution over arms.
        Args:
            feedback (float): Feedback signal.
            norm (float, optional): Normalization factor. Defaults to 1.0.
        """

        # Since this is non-stationary, subtract mean of previous self.window errors.
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-self.window :]

        # normalize
        feedback -= np.mean(self.error_buffer)
        if self.use_std and len(self.error_buffer) > 1:
            norm = np.std(self.error_buffer)
        feedback /= norm + 1e-4

        # update arm weights
        self.w[self.arm] *= self.decay
        self.w[self.arm] += self.lr * (feedback / max(np.exp(self.w[self.arm]), 0.0001))

        self.data.append(feedback)


class Bandit(object):
    """
    Utrzymuje 10 bandytów?
    """

    def __init__(
        self,
        arms: List = [0, 1],
        lr: float = 0.1,
        window: int = 5,
        decay: float = 0.9,
        init: float = 0.0,
        use_std: bool = True,
        num_seeds: int = 10,
    ) -> None:

        self.models = []
        self.num_seeds = num_seeds
        self.n_arms = len(arms)
        self.prev_rewards = np.zeros(num_seeds)
        for i in range(num_seeds):
            model = ExpWeights(arms=arms, lr=lr, window=window, decay=decay, init=init, use_std=use_std)
            self.models.append(model)

    def sample(self):
        arms = []
        for i in range(self.num_seeds):
            arm = self.models[i].sample()
            arms.append(arm)
        return np.stack(arms).astype(np.float64).reshape(self.num_seeds, 1)

    def get_probs(self):
        probs = []
        for i in range(self.num_seeds):
            prob = self.models[i].get_probs()
            probs.append(prob)
        return np.stack(probs).astype(np.float64)

    def update_prev_rewards(self, rewards):
        self.prev_rewards = rewards

    def update(self, rewards):
        rewards = np.squeeze(rewards)
        feedback_ = rewards - self.prev_rewards
        for i in range(self.num_seeds):
            self.models[i].update_dists(feedback=feedback_[i])
        self.update_prev_rewards(rewards)
