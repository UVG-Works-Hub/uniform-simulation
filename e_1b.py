import numpy as np
from scipy.stats import ks_2samp

# Datos continuos: alturas en centímetros
grupo1 = np.array([170, 172, 168, 169, 171, 173, 175, 174, 169, 170])
grupo2 = np.array([165, 167, 166, 168, 170, 172, 171, 169, 168, 167])

# Realizar la prueba KS
statistic, p_value = ks_2samp(grupo1, grupo2)

print(f'Estadístico KS: {statistic}')
print(f'Valor p: {p_value}')

# Interpretación
alpha = 0.05
if p_value < alpha:
    print('Rechazamos H₀: Las distribuciones son diferentes.')
else:
    print('No rechazamos H₀: Las distribuciones son iguales.')