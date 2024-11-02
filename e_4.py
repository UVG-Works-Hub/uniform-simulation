import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class LinearCongruentialGenerator:
    '''
    Linear Congruential Generator

    Parameters
    ----------
    seed : int
        The seed for the generator.
    a : int
        The multiplier.
    c : int
        The increment.
    m : int
        The modulus.
    '''

    def __init__(self, seed, a , c, m):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        '''
        Generate the next random number in the sequence.

        Returns
        -------
        int
            The next random number in the sequence
        '''
        self.state = (self.a * self.seed + self.c) % self.m # X_{n+1} = (a*X_n + c) mod m
        return self.state

    def generate_uniform(self, N):
        '''
        Generate N random numbers in the sequence.

        Parameters
        ----------
        N : int
            The number of random numbers to generate.
        '''
        numbers = np.empty(N)
        for i in range(N):
            numbers[i] = self.next() / self.m # U_n = X_n / m (Division by m to get a number between 0 and 1)
        return numbers

def perform_tests(samples):
    # Kolmogorov-Smirnov test for uniformity
    ks_stat, ks_p = stats.kstest(samples, 'uniform')

    # Chi-squared test for uniformity
    # Divide [0, 1) into 10 intervals
    observed, _ = np.histogram(samples, bins=10, range=(0, 1)) # Basically counting the number of samples in each interval
    expected = np.full(10, len(samples) / 10) # Expected number of samples in each interval

    chi2_stat, chi2_p = stats.chisquare(observed, expected)

    return {
        'KS Statistic': ks_stat,
        'KS p-value': ks_p,
        'Chi2 Statistic': chi2_stat,
        'Chi2 p-value': chi2_p
    }

def plot_hist(samples, title):
    '''
    Plot a histogram of the samples.

    Parameters
    ----------
    samples : np.array
        The samples to plot.
    title : str
        The title of the plot.
    '''
    plt.hist(samples, bins=10, range=(0, 1), edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('Frequency')
    plt.show()

def run_experiment(seed, a, c, m, N, experiment_num):
    '''
    Run an experiment with the given parameters.

    Parameters
    ----------
    seed : int
        The seed for the generator.
    a : int
        The multiplier.
    c : int
        The increment.
    m : int
        The modulus.
    N : int
        The number of samples to generate.
    experiment_num : int
        The number of the experiment
    '''
    print(f"\n--- Experiment {experiment_num} ---")
    print(f"Parameters: m={m}, a={a}, c={c}, N={N}")

    lgc = LinearCongruentialGenerator(seed, a, c, m)
    samples = lgc.generate_uniform(N)

    # Perform hypothesis tests
    results = perform_tests(samples)
    print("Hypothesis Test Results:")
    print(f"Kolmogorov-Smirnov Test: Statistic={results['KS Statistic']:.4f}, p-value={results['KS p-value']:.4f}")
    print(f"Chi-Square Test: Statistic={results['Chi2 Statistic']:.4f}, p-value={results['Chi2 p-value']:.4f}")

    # Plot histogram
    plot_hist(samples, f"Experiment {experiment_num} Histogram")

    return results

# Define experiment parameters
experiments = [
    {'seed': 1, 'a': 1103515245, 'c': 12345, 'm': 2**31, 'N': 10000},
    {'seed': 1, 'a': 75, 'c': 0, 'm': 2**16, 'N': 10000},
    {'seed': 1, 'a': 1664525, 'c': 1013904223, 'm': 2**32, 'N': 10000}
]

# Run experiments
results_all = []
for i, params in enumerate(experiments, 1):
    result = run_experiment(**params, experiment_num=i)
    results_all.append(result)
