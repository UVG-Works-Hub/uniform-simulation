import numpy as np
from scipy.stats import norm, chisquare, ks_2samp
import matplotlib.pyplot as plt
from collections import Counter

mu = 0.0  # Media
sigma = 1.0  # Desviación estándar
N = 100000  # Tamaño de la muestra
alpha = 0.05  # Nivel de significancia

np.random.seed(42)


def generate_empirical_sample_normal(mu, sigma, N):
    """
    Genera una muestra empírica usando la transformada inversa para la distribución Normal.

    Parameters:
        mu (float): Media de la distribución Normal.
        sigma (float): Desviación estándar de la distribución Normal.
        N (int): Tamaño de la muestra.

    Returns:
        np.ndarray: Muestra empírica generada.
    """
    U = np.random.uniform(0, 1, N)
    X = norm.ppf(U, loc=mu, scale=sigma)
    return X


def calculate_number_of_bins(data, method="sturges"):
    if method == "sturges":
        return int(np.ceil(np.log2(len(data)) + 1))
    elif method == "scott":
        std_dev = np.std(data)
        bin_width = 3.5 * std_dev / (len(data) ** (1 / 3))
        return int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    elif method == "fd":
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (len(data) ** (1 / 3))
        return int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    else:
        raise ValueError("Método desconocido. Usa 'sturges', 'scott' o 'fd'.")


def chisquare_test(theoretical_sample, empirical_sample, num_bins):
    min_edge = min(np.min(theoretical_sample), np.min(empirical_sample))
    max_edge = max(np.max(theoretical_sample), np.max(empirical_sample))
    bins = np.linspace(min_edge, max_edge, num_bins + 1)

    observed_counts, _ = np.histogram(empirical_sample, bins=bins)
    expected_probs = norm.cdf(bins[1:], loc=mu, scale=sigma) - norm.cdf(
        bins[:-1], loc=mu, scale=sigma
    )
    expected_counts = expected_probs * N

    while np.any(expected_counts < 5):
        min_count_index = np.argmin(expected_counts)
        if min_count_index < len(expected_counts) - 1:
            observed_counts[min_count_index + 1] += observed_counts[min_count_index]
            expected_counts[min_count_index + 1] += expected_counts[min_count_index]
        else:
            observed_counts[min_count_index - 1] += observed_counts[min_count_index]
            expected_counts[min_count_index - 1] += expected_counts[min_count_index]
        observed_counts = np.delete(observed_counts, min_count_index)
        expected_counts = np.delete(expected_counts, min_count_index)

    difference = N - np.sum(expected_counts)
    expected_counts[-1] += difference

    chi2_stat, p_value_chi2 = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    return chi2_stat, p_value_chi2


def ks_test(theoretical_sample, empirical_sample):
    ks_stat, p_value_ks = ks_2samp(theoretical_sample, empirical_sample)
    return ks_stat, p_value_ks


def plot_distributions(theoretical_sample, empirical_sample, bins):
    plt.figure(figsize=(12, 8))
    plt.hist(
        theoretical_sample,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Teórica (scipy.stats)",
        color="blue",
    )
    plt.hist(
        empirical_sample,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Empírica (Transformada Inversa)",
        color="orange",
    )
    plt.xlabel("Valor de X")
    plt.ylabel("Densidad de Probabilidad")
    plt.title("Comparación de Distribuciones Normales Teórica y Empírica")
    plt.legend()
    plt.tight_layout()
    plt.savefig("normal_distribution_comparison.png", dpi=300)
    print("Gráfico guardado como 'normal_distribution_comparison.png'")


def main():

    empirical_sample = generate_empirical_sample_normal(mu, sigma, N)
    theoretical_sample = norm.rvs(loc=mu, scale=sigma, size=N)
    method = "scott"
    num_bins = calculate_number_of_bins(empirical_sample, method=method)

    plot_distributions(theoretical_sample, empirical_sample, num_bins)

    chi2_stat, p_value_chi2 = chisquare_test(
        theoretical_sample, empirical_sample, num_bins
    )
    ks_stat, p_value_ks = ks_test(theoretical_sample, empirical_sample)

    print(
        f"Prueba de Chi-Cuadrado:\n  - Estadístico Chi-Cuadrado: {chi2_stat:.4f}\n  - Valor p: {p_value_chi2:.4f}"
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
