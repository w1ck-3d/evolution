import numpy as np


class WrightFisher:
    def __init__(
        self,
        N: int = 100,  # population size
        a: float = 0.5,  # initial fraction of alleles of type a
        s: float = 0.0,  # selection coefficient
        mu_1: float = 0.0,  # probability of mutation from a to A
        mu_2: float = 0.0,  # probability of mutation from A to a
    ) -> None:
        assert 0.0 <= a <= 1.0
        assert 0.0 <= mu_1 <= 1.0
        assert 0.0 <= mu_2 <= 1.0

        self.a = round(N * a)
        self.N = N
        self.s = s
        self.mu_1 = mu_1
        self.mu_2 = mu_2

        self.t = 0  # generation
        self.history = [self.a]

    @property
    def A(self):
        return self.N - self.a

    def __str__(self) -> str:
        return f"Wright-Fisher (a: {self.a}, A: {self.A})"

    def step(self):
        self.t += 1
        k = self.a
        s = self.s
        N = self.N
        mu_1 = self.mu_1
        mu_2 = self.mu_2
        psi = (k * (1 + s) * (1 - mu_1) + (N - k) * mu_2) / (k * (1 + s) + N - k)
        self.a = round(np.random.binomial(self.N, psi))
        self.history.append(self.a)

    def get_history(self):
        return self.history


SIMULATION_COUNT = 30
GENERATION_COUNT = 100
N = 1000
initial_prob_a = 0.1  # initial probability of allele a
s = 0.2  # selection coefficient
u = 0.1  # probability of mutation from a to A

simulations = []
for _ in range(SIMULATION_COUNT):
    wf = WrightFisher(N, initial_prob_a, s, u)
    for _ in range(GENERATION_COUNT):
        wf.step()
    simulations.append(wf.get_history())
