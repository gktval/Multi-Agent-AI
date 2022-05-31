import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, x0=1.0, mu=0., theta=1.0, sigma=0.2):
        """Initialize parameters and noise process."""
        self.theta = theta
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.x = 0
        self.W = 0
        self.t = 0

    def reset(self, sigma):
        """Reset the internal state (= noise) to mean (mu)."""
        self.W = 0
        self.t = 0
        self.x = self.mu
        self.sigma = sigma

    def sample(self,dt=1e-2):
        """Update internal state and return it as a noise sample."""
        
        ex = np.exp(-self.theta * (self.t + dt))
        self.W += np.sqrt(np.exp(2 * self.theta * (self.t + dt)) - np.exp(2 * self.theta * self.t)) * \
                  np.random.randn() / np.sqrt(2 * self.theta)
        # self.x = self.x0 * ex + self.mu * (1 - ex) + self.sigma * ex * self.W
        self.x = self.mu * (1 - ex) + self.sigma * ex * self.W
        self.t += dt
        return self.x