import numpy as np
import matplotlib.pyplot as plt


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
        if self.is_a_fixed() or self.is_a_lost():
            self.history.append(self.a)  # nothing changes
            return

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

    def is_a_fixed(self):
        return self.a == self.N

    def is_a_lost(self):
        return self.a == 0


SIMULATION_COUNT = 100
N = 10  # population size
initial_prob_a = 1.0 / N  # initial probability of allele a
s = 0.0  # selection coefficient
u = 0.0  # probability of mutation from a to A

# === START OVERWRITE
# N, initial_prob_a, s, u
ALPHA_5 = (100, 0.3, 0.05, 0.0)  # expecected fix prob: 0.95, actual: 0.73
NEARLY_NEUTRAL = (100, 1.0 / 100, 0.0001, 0.0)  # fix prob: 1/N
BENEFICIAL = (100, 1.0 / 100, 0.001, 0.0)  # expected fix prob: 2*s
DELETERIOUS = (100, 1.0 / 100, -0.001, 0.0)  # expected fix prob: 1/N

N, initial_prob_a, s, u = DELETERIOUS
# === END OVERWRITE

simulations = []
max_generation_count = 0
fixation_count = 0
for _ in range(SIMULATION_COUNT):
    wf = WrightFisher(N, initial_prob_a, s, u)
    while not wf.is_a_lost() and not wf.is_a_fixed():
        wf.step()
    simulations.append(wf.get_history())
    if wf.is_a_fixed():
        fixation_count += 1
    max_generation_count = max(max_generation_count, wf.t)

for sim in simulations:
    sim.extend([sim[-1]] * (max_generation_count - len(sim) + 1))

x = list(range(1, max_generation_count + 2))

print("Fixation probability:", fixation_count / SIMULATION_COUNT)

for sim in simulations:
    plt.plot(x, sim)

plt.title("Wright-Fisher Model With Selection and Mutation")
plt.xlabel("generation")
plt.ylabel("# a alleles")
plt.xlim([0, max_generation_count])
plt.ylim([0, N])
# green_colors = plt.cm.get_cmap("Greens", SIMULATION_COUNT * 2)

plt.savefig("wright_fisher.png")
