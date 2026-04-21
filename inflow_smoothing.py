import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.interpolate import CubicSpline

patient_numbers = [5, 8, 11]
for patient_number in patient_numbers:
        
    og_dir = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models"
    simulation_file = os.path.join(og_dir, "zeroD_simulation")
    saving_file = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Simulations/fine/{patient_number:01d}_zeroD_opt"

    flow_files = {
        "LICA": os.path.join(simulation_file, "LICA.dat"),
        "RICA": os.path.join(simulation_file, "RICA.dat"),
        "LVA": os.path.join(simulation_file, "LVA.dat"),
        "RVA": os.path.join(simulation_file, "RVA.dat")
    }

    plt.figure(figsize=(12, 8))

    # Parámetros para el header
    N_POINTS_OUT = 100  # Reducimos a 100 puntos para que el solver sea más eficiente
    FOURIER_MODES = 10

    for i, (label, path) in enumerate(flow_files.items()):
        if os.path.exists(path):
            data = np.loadtxt(path, skiprows=1)
            sort_idx = np.argsort(data[:, 0])
            time_pts = data[sort_idx, 0]
            flow_pts = np.abs(data[sort_idx, 1])

            # ... (Tu lógica de puntos falsos y aumentados se mantiene igual)
            t_penult, t_last = time_pts[-2], time_pts[-1]
            f_penult, f_last = flow_pts[-2], flow_pts[-1]
            n_puntos_falsos = 2
            t_falsos = np.linspace(t_penult, t_last, n_puntos_falsos + 2)[1:-1]
            f_falsos = f_penult + (t_falsos - t_penult) * (f_last - f_penult) / (t_last - t_penult)
            
            time_pts_augmented = np.concatenate([time_pts[:-1], t_falsos, [t_last]])
            flow_pts_augmented = np.concatenate([flow_pts[:-1], f_falsos, [f_last]])

            # Spline
            try:
                cs = CubicSpline(time_pts_augmented, flow_pts_augmented, bc_type='not-a-knot')
            except ValueError:
                cs = CubicSpline(time_pts_augmented, flow_pts_augmented, bc_type='clamped')
            
            # Generamos los puntos de salida finales
            time_smooth = np.linspace(time_pts.min(), time_pts.max(), N_POINTS_OUT)
            flow_smooth = cs(time_smooth)
            
            # --- NUEVA SECCIÓN DE GUARDADO CON HEADER ---
            output_path_3d = os.path.join(saving_file, f"{label}_3d_smooth.dat") # Cambiado a .flow
            output_path_0d = os.path.join(simulation_file, f"{label}_0d_smooth.dat")
            
            with open(output_path_3d, 'w') as f:
                # Escribimos el header: N_puntos, N_fourier
                f.write(f"{N_POINTS_OUT},{FOURIER_MODES}\n")
                # Escribimos los datos (SimVascular prefiere flujos negativos para Inlets)
                for t, fl in zip(time_smooth, flow_smooth):
                    f.write(f"{t:.6e}\t{-abs(fl):.6e}\n") # Forzamos negativo por convención de entrada
            
            # with open(output_path_0d, 'w') as f:
            #     # Escribimos los datos (SimVascular prefiere flujos negativos para Inlets)
            #     for t, fl in zip(time_smooth, flow_smooth):
            #         f.write(f"{t:.6e}\t{-abs(fl):.6e}\n")

            # --- PLOT ---
            plt.subplot(2, 2, i+1)
            plt.scatter(time_pts_augmented, flow_pts_augmented, color='red', s=20, label='Puntos Clínicos')
            plt.plot(time_smooth, flow_smooth, color='blue', label='Cubic Spline', linewidth=1.5)
            plt.title(f"Inlet: {label} (Saved as .flow)")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Flujo (mm³/s)")
            plt.grid(True, alpha=0.3)
            plt.legend()

    plt.tight_layout()
    plt.show()