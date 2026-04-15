import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models"
simulation_file = os.path.join(og_dir, "zeroD_simulation")

# Diccionario para automatizar la carga
flow_files = {
    "LICA": os.path.join(simulation_file, "LICA.dat"),
    "RICA": os.path.join(simulation_file, "RICA.dat"),
    "LVA": os.path.join(simulation_file, "LVA.dat"),
    "RVA": os.path.join(simulation_file, "RVA.dat")
}

plt.figure(figsize=(10, 6))
for label, path in flow_files.items():
    if os.path.exists(path):
        data = np.loadtxt(path)
        time_pts = data[:, 0]
        flow_pts = np.abs(data[:, 1])
        
        # 1. Crear el Spline Cúbico
        # bc_type='periodic' asegura que el final del ciclo conecte suave con el inicio
        cs = CubicSpline(time_pts, flow_pts, bc_type='periodic')
        
        # 2. Crear un nuevo eje de tiempo con más puntos (ej. 10 veces más)
        time_smooth = np.linspace(time_pts.min(), time_pts.max(), len(time_pts) * 10)
        flow_smooth = cs(time_smooth)

        output_path = os.path.join(simulation_file, f"{label}_smooth.dat")
        
        # Stack horizontal de las dos columnas
        new_data = np.column_stack((time_smooth, flow_smooth))
        
        # Guardar con formato científico (como tus originales) y separador de tabulador
        np.savetxt(output_path, new_data, delimiter='\t', fmt='%e')
        
        # Graficar
        plt.scatter(time_pts, flow_pts, alpha=1, s=30, label=f"{label} (Nodos)")
        plt.plot(time_smooth, flow_smooth, label=f"{label} (Spline)", linewidth=2)
        plt.plot(time_pts, flow_pts, label=f"{label} (Spline)", linewidth=2)
        
    else:
        print(f"Advertencia: No se encontró el archivo {path}")

plt.title("Interpolación por Splines Cúbicos (Preservación de Picos)")
plt.xlabel("Time (s)")
plt.ylabel("Flow (mm³/s)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()