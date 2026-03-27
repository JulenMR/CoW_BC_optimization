import pysvzerod
import json
import pandas as pd
import importlib.util
import sys
import os
import matplotlib.pyplot as plt

# 1. Ruta exacta al archivo postprocessing.py en tu repositorio
# Verifica en tu terminal si esta ruta es correcta: ls ~/svZeroDSolver/python/pysvzerod/postprocessing.py
example_file = "/home/julenmr/Downloads/SVAortofemoral/ROMSimulations/zerotwo/solver_0d.json"

# 1. Cargar el JSON generado por SV
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines"
# Procesar ambos ejemplos
centerline = os.path.join(og_dir, "easy_example", "cow_full_final.vtp"), 
json_file_easy = os.path.join(og_dir, "easy_example", "zeroD_script.json")
json_file_meduim = os.path.join(og_dir, "medium_example", "zeroD_script.json")
json_file_full = os.path.join(og_dir, "zeroD_script_full.json")
model_config = json.load(open(json_file_full))

# 2. Inicializar y correr el solver
print(" Ejecutando simulación 0D...")
solver = pysvzerod.Solver(model_config)
solver.run()

# 3. Obtener el DataFrame de resultados
df = solver.get_full_result()

# 4. Mostrar las primeras filas y estadísticas básicas
print("\n--- Vista previa de los resultados ---")
print(df.head())
print(df["name"].unique())


import matplotlib.pyplot as plt

def plot_custom_0d_results(df):
    plt.figure(figsize=(10, 6))
    
    # 1. Extraemos los datos para cada segmento específico
    # Usamos los nombres exactos de tu JSON: branch0_seg0 y branch2_seg0
    b0 = df[df['name'] == 'branch0']
    b2 = df[df['name'] == 'branch2']
    b1 = df[df['name'] == 'branch1'] # El tronco (entrada)

    # 2. Plotear Presión de Salida (pressure_out) para los dos últimos
    if not b0.empty:
        plt.plot(b0['time'], b0['pressure_out'], label='Presión Salida Rama 0', linewidth=2)
    if not b2.empty:
        plt.plot(b2['time'], b2['pressure_out'], label='Presión Salida Rama 2', linewidth=2)
    
    # 3. Opcional: Presión de Entrada en el tronco para ver la caída total
    if not b1.empty:
        plt.plot(b1['time'], b1['pressure_in'], '--', label='Presión Entrada Tronco (b1)', alpha=0.6)

    # Configuración estética
    plt.title('Presión en los Segmentos de Salida de la Bifurcación')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Presión (Baryes)') # Recuerda: mmHg = Baryes / 1333.2
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Mostrar valores finales por consola (útil para debug)
    if not b0.empty and not b2.empty:
        print(f"--- Valores Finales (Estado Estacionario) ---")
        print(f"P_out Rama 0: {b0['pressure_out'].iloc[-1]:.2f}")
        print(f"P_out Rama 2: {b2['pressure_out'].iloc[-1]:.2f}")

    plt.show()

plot_custom_0d_results(df)
