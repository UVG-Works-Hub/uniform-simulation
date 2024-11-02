'''
@Author: Samuel Chamalé
@Date: November 2024
@Description: Linear Congruential Generator (LCG) and Hypothesis Testing
'''

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
    bits : tuple, optional
        Tuple indicating (high_bit, low_bit) to extract for output.
        If None, use the full range [0, m).
    '''

    def __init__(self, seed, a , c, m, bits=None):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m
        self.bits = bits  # e.g., (30, 16) to extract bits 30..16

    def next(self):
        '''
        Generate the next random number in the sequence.

        Returns
        -------
        int
            The next random number in the sequence
        '''
        self.state = (self.a * self.state + self.c) % self.m # X_{n+1} = (a*X_n + c) mod m
        if self.bits:
            high_bit, low_bit = self.bits
            # Extract bits from high_bit down to low_bit
            bit_length = high_bit - low_bit + 1
            # Shift right by low_bit and mask
            extracted = (self.state >> low_bit) & ((1 << bit_length) - 1)
            return extracted
        else:
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
            x = self.next()

            # U_n = X_n / m (Division by m to get a number between 0 and 1)
            if self.bits:
                # Normalize based on the number of bits extracted
                max_val = (1 << (self.bits[0] - self.bits[1] + 1)) - 1
                numbers[i] = x / (max_val + 1)
            else:
                numbers[i] = self.next() / self.m

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
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=10, range=(0,1), edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def run_experiment(seed, a, c, m, bits, N, experiment_num, name):
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
    bits : tuple
        Tuple indicating (high_bit, low_bit) to extract for output.
        If None, use the full range [0, m).
    N : int
        The number of samples to generate.
    experiment_num : int
        The number of the experiment
    '''
    print(f"\n--- Experiment {experiment_num}: {name} ---")
    print(f"Parameters: m={m}, a={a}, c={c}, N={N}")
    if bits:
        print(f"Output Bits: {bits[0]}..{bits[1]}")

    lcg = LinearCongruentialGenerator(seed, a, c, m, bits)
    samples = lcg.generate_uniform(N)

    # Perform hypothesis tests
    results = perform_tests(samples)
    print("Hypothesis Test Results:")
    print(f"Kolmogorov-Smirnov Test: Statistic={results['KS Statistic']:.4f}, p-value={results['KS p-value']:.4f}")
    print(f"Chi-Square Test: Statistic={results['Chi2 Statistic']:.4f}, p-value={results['Chi2 p-value']:.4f}")

    # Plot histogram
    plot_hist(samples, f"Histogram of LCG Samples - Experiment {experiment_num}: {name}")

    return results

# Define experiments
# Parameters obtained from https://en.wikipedia.org/wiki/Linear_congruential_generator
experiments = [
    {
        'name': 'glibc (GNU C Library)',
        'seed': 1,
        'a': 1103515245,
        'c': 12345,
        'm': 2**31,
        'bits': None,  # Use all bits
        'N': 10000
    },
    {
        'name': 'Microsoft Visual C/C++',
        'seed': 1,
        'a': 214013,
        'c': 2531011,  # 0x343FD
        'm': 2**31,
        'bits': (30, 16),  # Output bits 30..16
        'N': 10000
    },
    {
        'name': "Java's java.util.Random",
        'seed': 1,
        'a': 25214903917,  # 0x5DEECE66D
        'c': 11,
        'm': 2**48,
        'bits': (47, 16),  # Output bits 47..16
        'N': 10000
    }
]

# Run experiments
results_all = []
for i, params in enumerate(experiments, 1):
    result = run_experiment(
        seed=params['seed'],
        a=params['a'],
        c=params['c'],
        m=params['m'],
        bits=params['bits'],
        N=params['N'],
        experiment_num=i,
        name=params['name']
    )
    results_all.append(result)
