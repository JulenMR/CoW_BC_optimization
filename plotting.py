import numpy as np
import matplotlib.pyplot as plt

def generate_vascular_flow(t, peak_q, baseline_q, hr_period):
    """
    Crea una onda de flujo sintética basada en parámetros fisiológicos.
    """
    # 1. Componente Sistólica (Pico rápido)
    systole = peak_q * np.exp(-((t - 0.12 * hr_period)**2) / (2 * (0.045 * hr_period)**2))
    
    # 2. Componente Dicrota (El pequeño rebote post-sístole)
    dicrotic = (peak_q * 0.35) * np.exp(-((t - 0.42 * hr_period)**2) / (2 * (0.07 * hr_period)**2))
    
    # 3. Componente Diastólica (Flujo constante de fondo con decaimiento)
    diastole = baseline_q * (1 + 0.2 * np.exp(-t / (0.5 * hr_period)))
    
    # Combinación (negativa porque es Inlet en SimVascular)
    flow = -(systole + dicrotic + diastole)
    
    # Forzar periodicidad exacta para el solver
    flow[-1] = flow[0]
    return flow

# Configuración
period = 1.022
t_new = np.linspace(0, period, 100) # 100 muestras

# Configuración de valores basada en tus .dat originales
configs = {
    "LICA": {"peak": 4800, "base": 2400},
    "RICA": {"peak": 3400, "base": 1400},
    "LVA":  {"peak": 1550, "base": 650},
    "RVA":  {"peak": 1150, "base": 480}
}

plt.figure(figsize=(10, 6))

for name, val in configs.items():
    q_syn = generate_vascular_flow(t_new, val['peak'], val['base'], period)
    
    # Guardar archivo .dat con alta resolución
    np.savetxt(f"{name}_synthetic.dat", np.column_stack((t_new, q_syn)), 
               fmt='%.6e', delimiter='\t', header="100,10", comments='')
    
    plt.plot(t_new, q_syn, label=f"{name} (Sintética)")

plt.title("Inflows Sintéticos de Alta Resolución (100 puntos)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Flujo (mm³/s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()