import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.interpolate import CubicSpline

og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models"
simulation_file = os.path.join(og_dir, "zeroD_simulation")

flow_files = {
    "LICA": os.path.join(simulation_file, "LICA.dat"),
    "RICA": os.path.join(simulation_file, "RICA.dat"),
    "LVA": os.path.join(simulation_file, "LVA.dat"),
    "RVA": os.path.join(simulation_file, "RVA.dat")
}

plt.figure(figsize=(12, 8))

for i, (label, path) in enumerate(flow_files.items()):
    if os.path.exists(path):
        data = np.loadtxt(path, skiprows=1)
        sort_idx = np.argsort(data[:, 0])
        time_pts = data[sort_idx, 0]
        flow_pts = np.abs(data[sort_idx, 1])
        
        # --- GENERACIÓN DE PUNTOS FALSOS (Lineal entre penúltimo y último) ---
        # Tomamos los dos últimos puntos reales
        t_penult, t_last = time_pts[-2], time_pts[-1]
        f_penult, f_last = flow_pts[-2], flow_pts[-1]
        
        n_puntos_falsos = 2
        t_falsos = np.linspace(t_penult, t_last, n_puntos_falsos + 2)[1:-1]
        # Interpolación lineal: f = f1 + (t - t1) * (f2 - f1) / (t2 - t1)
        f_falsos = f_penult + (t_falsos - t_penult) * (f_last - f_penult) / (t_last - t_penult)
        
        # Reconstruimos los arrays insertando los puntos falsos
        time_pts_augmented = np.concatenate([time_pts[:-1], t_falsos, [t_last]])
        flow_pts_augmented = np.concatenate([flow_pts[:-1], f_falsos, [f_last]])
        # ----------------------------------------------------------------------

        # Ahora aplicamos el Spline sobre los datos aumentados
        try:
            # Al haber una transición lineal, 'not-a-knot' suele ser más estable que 'periodic'
            cs = CubicSpline(time_pts_augmented, flow_pts_augmented, bc_type='not-a-knot')
        except ValueError:
            cs = CubicSpline(time_pts_augmented, flow_pts_augmented, bc_type='clamped')
        
        time_smooth = np.linspace(time_pts.min(), time_pts.max(), 1000)
        flow_smooth = cs(time_smooth)
        
        # 4. Guardar resultados
        output_path = os.path.join(simulation_file, f"{label}_smooth.dat")
        np.savetxt(output_path, np.column_stack((time_smooth, flow_smooth)), delimiter='\t', fmt='%e')
        
        # --- PLOT ---
        plt.subplot(2, 2, i+1)
        plt.scatter(time_pts_augmented, flow_pts_augmented, color='red', s=20, label='Puntos Clínicos')
        plt.plot(time_smooth, flow_smooth, color='blue', label='Cubic Spline', linewidth=1.5)
        plt.title(f"Inlet: {label}")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Flujo (mm³/s)")
        plt.grid(True, alpha=0.3)
        plt.legend()

plt.tight_layout()
plt.show()