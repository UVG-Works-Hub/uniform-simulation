import numpy as np
from scipy.stats import chi2_contingency

# Datos: Filas = Grupos, Columnas = Sabores
# Grupo 1: [30, 10, 20] (Sabores A, B, C)
# Grupo 2: [20, 20, 10]
tabla_contingencia = np.array([
    [30, 10, 20],
    [20, 20, 10]
])

# Realizar la prueba de Chi Cuadrado
chi2, p, dof, ex = chi2_contingency(tabla_contingencia)

print(f"Chi-cuadrado: {chi2}")
print(f"Valor p: {p}")
print(f"Grados de libertad: {dof}")
print("Frecuencias esperadas:")
print(ex)

# Interpretación
alpha = 0.05
if p < alpha:
    print("Rechazamos la hipótesis nula: Hay diferencias significativas entre las distribuciones.")
else:
    print("No rechazamos la hipótesis nula: No hay diferencias significativas entre las distribuciones.")
