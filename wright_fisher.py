import random


class WrightFisher:
    def __init__(
        self,
        a: int = 3,  # initial amount of alleles of type a
        N: int = 10,  # population size
        s: float = 0.0,  # selection coefficient
        mu_1: float = 0.0,  # probability of mutation from a to A
        mu_2: float = 0.0,  # probability of mutation from A to a
    ) -> None:
        assert 0 <= a <= N
        assert 0.0 <= mu_1 <= 1.0
        assert 0.0 <= mu_2 <= 1.0

        self.t = 0  # generation
        self.a = a
        self.N = N
        self.s = s
        self.mu_1 = mu_1
        self.mu_2 = mu_2

    @property
    def A(self):
        return self.N - self.a

    def __str__(self) -> str:
        return f"Wright-Fisher (a: {self.a}, A: {self.A})"

    def step(self):
        self.t += 1

        new_a = 0
        prob_of_a_sampled = (
            self.a * (1 + self.s) * (1 - self.mu_1) + (self.N - self.a) * self.mu_2
        ) / (self.a * (1 + self.s) + self.N - self.a)
        for _ in range(self.N):
            if random.random() < prob_of_a_sampled:
                new_a += 1
        self.a = new_a


NUM_STEPS = 10

wf = WrightFisher()

for _ in range(NUM_STEPS):
    wf.step()
    print(wf)
