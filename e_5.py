'''
@Author: Samuel Chamalé
@Date: November 2024
@Description: Mersenne Twister (MT19937) Implementation in Python with Hypothesis Testing
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class MersenneTwister:
    '''
    Mersenne Twister (MT19937) Pseudorandom Number Generator

    Parameters
    ----------
    seed : int
        The seed for the generator.

    References
    ----------
    [1] Makoto Matsumoto and Takuji Nishimura. 1998. Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator. ACM Trans. Model. Comput. Simul. 8, 1 (Jan. 1998), 3–30. https://doi.org/10.1145/272991.272995
    [2] https://en.wikipedia.org/wiki/Mersenne_Twister
    '''

    # Define constants
    w, n, m, r = 32, 624, 397, 31
    a = 0x9908B0DF
    u, d = 11, 0xFFFFFFFF
    s, b = 7, 0x9D2C5680
    t, c = 15, 0xEFC60000
    l = 18
    f = 1812433253

    # Bitmasks
    lower_mask = (1 << r) - 1  # The binary number of r 1's
    upper_mask = (~lower_mask) & 0xFFFFFFFF  # The binary number of w-r 1's followed by r 0's

    def __init__(self, seed):
        '''
        Initialize the generator from a seed.

        Parameters
        ----------
        seed : int
            The seed for the generator.
        '''
        self.mt = [0] * self.n  # Initialize the state array
        self.index = self.n  # Initialize index to n to trigger twist on first use
        self.mt[0] = seed
        for i in range(1, self.n):
            temp = self.f * (self.mt[i - 1] ^ (self.mt[i - 1] >> (self.w - 2))) + i
            self.mt[i] = temp & 0xFFFFFFFF  # Get 32 least significant bits

    def twist(self):
        '''
        Generate the next n values from the series x_i.
        '''
        for i in range(self.n):
            x = (self.mt[i] & self.upper_mask) + (self.mt[(i + 1) % self.n] & self.lower_mask)
            xA = x >> 1
            if x % 2 != 0:  # Lowest bit of x is 1
                xA ^= self.a
            self.mt[i] = self.mt[(i + self.m) % self.n] ^ xA
        self.index = 0

    def extract_number(self):
        '''
        Extract a tempered pseudorandom number based on the index-th value,
        calling twist() every n numbers.

        Returns
        -------
        int
            A pseudorandom number.
        '''
        if self.index >= self.n:
            self.twist()

        y = self.mt[self.index]
        y ^= (y >> self.u) & self.d
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= (y >> self.l)

        self.index += 1
        return y & 0xFFFFFFFF  # Return 32-bit integer

    def generate_uniform(self, N):
        '''
        Generate N uniform random numbers in [0,1).

        Parameters
        ----------
        N : int
            The number of random numbers to generate.

        Returns
        -------
        np.array
            Array of N uniform random numbers in [0,1).
        '''
        numbers = np.empty(N)
        for i in range(N):
            num = self.extract_number()
            numbers[i] = num / 4294967296.0  # Divide by 2^32
        return numbers

def perform_tests(samples):
    """
    Perform Kolmogorov-Smirnov and Chi-Square tests for uniformity.

    Parameters
    ----------
    samples : np.array
        The array of generated uniform random numbers.

    Returns
    -------
    dict
        Dictionary containing test statistics and p-values.
    """
    # Kolmogorov-Smirnov test for uniformity
    ks_stat, ks_p = stats.kstest(samples, 'uniform')

    # Chi-squared test for uniformity
    # Divide [0, 1) into 10 intervals
    observed, _ = np.histogram(samples, bins=10, range=(0, 1))
    expected = np.full(10, len(samples) / 10)
    chi2_stat, chi2_p = stats.chisquare(observed, expected)

    return {
        'KS Statistic': ks_stat,
        'KS p-value': ks_p,
        'Chi2 Statistic': chi2_stat,
        'Chi2 p-value': chi2_p
    }

def plot_hist(samples, title):
    """
    Plot a histogram of the samples.

    Parameters
    ----------
    samples : np.array
        The samples to plot.
    title : str
        The title of the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=10, range=(0,1), edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def run_experiment(seed, N, experiment_num, name):
    """
    Run an experiment with the Mersenne Twister generator.

    Parameters
    ----------
    seed : int
        The seed for the generator.
    N : int
        The number of samples to generate.
    experiment_num : int
        The number of the experiment.
    name : str
        The name of the experiment.

    Returns
    -------
    dict
        Dictionary containing test results.
    """
    print(f"\n--- Experiment {experiment_num}: {name} ---")
    print(f"Parameters: Seed={seed}, N={N}")

    # Initialize Mersenne Twister generator with the given seed
    mt = MersenneTwister(seed)

    # Generate N uniform random samples in [0, 1)
    samples = mt.generate_uniform(N)

    # Perform hypothesis tests
    results = perform_tests(samples)
    print("Hypothesis Test Results:")
    print(f"Kolmogorov-Smirnov Test: Statistic={results['KS Statistic']:.4f}, p-value={results['KS p-value']:.4f}")
    print(f"Chi-Square Test: Statistic={results['Chi2 Statistic']:.4f}, p-value={results['Chi2 p-value']:.4f}")

    # Plot histogram
    plot_hist(samples, f"Histogram of MT19937 Samples - Experiment {experiment_num}: {name}")

    return results

# Define experiments with different seeds
experiments = [
    {
        'name': 'Mersenne Twister - Seed 1',
        'seed': 1,
        'N': 10000
    },
    {
        'name': 'Mersenne Twister - Seed 12345',
        'seed': 12345,
        'N': 10000
    },
    {
        'name': 'Mersenne Twister - Seed 987654321',
        'seed': 987654321,
        'N': 10000
    }
]

# Run experiments
results_all_mt = []
for i, params in enumerate(experiments, 1):
    result = run_experiment(
        seed=params['seed'],
        N=params['N'],
        experiment_num=i,
        name=params['name']
    )
    results_all_mt.append(result)
