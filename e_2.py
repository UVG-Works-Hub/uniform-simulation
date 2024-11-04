import numpy as np
from scipy.stats import geom, chisquare, ks_2samp
import matplotlib.pyplot as plt
from collections import Counter

p = 0.25  # Probabilidad de éxito
N = 10000  # Tamaño de la muestra
alpha = 0.05  # Nivel de significancia
np.random.seed(42)


def generate_empirical_sample(p, N):
    """
    Genera una muestra empírica usando la transformada inversa para la distribución Geométrica.

    Parameters:
        p (float): Probabilidad de éxito.
        N (int): Tamaño de la muestra.

    Returns:
        np.ndarray: Muestra empírica generada.
    """
    U = np.random.uniform(0, 1, N)
    X = np.floor(np.log(1 - U) / np.log(1 - p)) + 1
    return X.astype(int)


def chisquare_test(theoretical_sample, empirical_sample):
    """
    Prueba de Chi-Cuadrado para comparar dos muestras.

    Parameters:
        theoretical_sample (np.ndarray): Muestra teórica.
        empirical_sample (np.ndarray): Muestra empírica.

    Returns:
        tuple: (Estadístico Chi-Cuadrado, p-valor de la prueba)
    """
    count_theoretical = Counter(theoretical_sample)
    count_empirical = Counter(empirical_sample)

    values = sorted(set(count_theoretical.keys()).union(count_empirical.keys()))
    max_x = max(values)

    observed = []
    expected = []

    for x in values:
        if x < max_x:
            observed.append(count_empirical.get(x, 0))
            expected.append(geom.pmf(x, p) * N)
        elif x == max_x:
            observed_sum = count_empirical.get(x, 0)
            expected_sum = geom.pmf(x, p) * N + geom.sf(x, p) * N
            observed.append(observed_sum)
            expected.append(expected_sum)

    difference = N - sum(expected)
    expected[-1] += difference

    chi2_stat, p_value_chi2 = chisquare(f_obs=observed, f_exp=expected)

    return chi2_stat, p_value_chi2


def ks_test(theoretical_sample, empirical_sample):
    """
    Prueba de Kolmogorov-Smirnov para comparar dos muestras.

    Parameters:
        theoretical_sample (np.ndarray): Muestra teórica.
        empirical_sample (np.ndarray): Muestra empírica.

    Returns:
        tuple: (Estadístico KS, p-valor de la prueba)
    """
    ks_stat, p_value_ks = ks_2samp(theoretical_sample, empirical_sample)

    return ks_stat, p_value_ks


def plot_distributions(theoretical_sample, empirical_sample):
    """
    Grafica las distribuciones teórica y empírica.

    Parameters:
        theoretical_sample (np.ndarray): Muestra teórica.
        empirical_sample (np.ndarray): Muestra empírica.
    """
    count_theoretical = Counter(theoretical_sample)
    count_empirical = Counter(empirical_sample)

    values = sorted(set(count_theoretical.keys()).union(count_empirical.keys()))

    freq_theoretical = [count_theoretical.get(x, 0) / N for x in values]
    freq_empirical = [count_empirical.get(x, 0) / N for x in values]

    plt.figure(figsize=(12, 8))

    plt.bar(
        [x - 0.2 for x in values],
        freq_theoretical,
        width=0.4,
        label="Theoretical (scipy)",
        alpha=0.7,
        color="blue",
    )

    plt.bar(
        [x + 0.2 for x in values],
        freq_empirical,
        width=0.4,
        label="Empirical (Inverse Transform)",
        alpha=0.7,
        color="orange",
    )

    plt.xlabel("Value of X")
    plt.ylabel("Relative Frequency")
    plt.title("Comparison of Theoretical and Empirical Geometric Distributions")
    plt.xticks(values)
    plt.legend()
    plt.tight_layout()
    plt.savefig("geometric_distribution_comparison.png", dpi=300)
    print("Gráfico guardado como 'geometric_distribution_comparison.png'")


def main():

    empirical_sample = generate_empirical_sample(p, N)
    theoretical_sample = geom.rvs(p, size=N)

    plot_distributions(theoretical_sample, empirical_sample)

    chi2_stat, p_value_chi2 = chisquare_test(theoretical_sample, empirical_sample)
    ks_stat, p_value_ks = ks_test(theoretical_sample, empirical_sample)

    print(
        f"\nPrueba de Chi-Cuadrado:\n  - Estadístico Chi-Cuadrado: {chi2_stat:.4f}\n  - Valor p: {p_value_chi2:.4f}"
    )
    if p_value_chi2 < alpha:
        print(
            "  - Conclusión: Rechazamos la hipótesis nula. Las muestras no provienen de la misma distribución."
        )
    else:
        print(
            "  - Conclusión: No se rechaza la hipótesis nula. Las muestras podrían provenir de la misma distribución."
        )

    print(
        f"\nPrueba de Kolmogorov-Smirnov:\n  - Estadístico KS: {ks_stat:.4f}\n  - Valor p: {p_value_ks:.4f}"
    )
    if p_value_ks < alpha:
        print(
            "  - Conclusión: Rechazamos la hipótesis nula. Las muestras no provienen de la misma distribución."
        )
    else:
        print(
            "  - Conclusión: No se rechaza la hipótesis nula. Las muestras podrían provenir de la misma distribución."
        )


if __name__ == "__main__":
    main()
