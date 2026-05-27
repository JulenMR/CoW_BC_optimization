import os
import re
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def check_mass_continuity_perfect(vtu_folder, surfaces_dir, inlets, outlets, time_step_size=0.001022, saved_every=20):
    print("Loading cap surface masks and calculating geometry invariants...")
    all_masks = {}
    
    for name in (inlets + outlets):
        path = os.path.join(surfaces_dir, f"cap_{name}.vtp")
        if os.path.exists(path):
            # Cargamos la superficie original limpia
            all_masks[name] = pv.read(path)
        else:
            print(f"[WARNING] Surface file not found for: {name}")

    all_files = [f for f in os.listdir(vtu_folder) if f.endswith('.vtu')]
    all_files.sort(key=natural_sort_key)
    vtu_files = [os.path.join(vtu_folder, f) for f in all_files]

    times = []
    total_inflow_list = []
    total_outflow_list = []
    mass_error_list = []
    time_step = time_step_size * saved_every

    for step_idx, vtu_path in enumerate(vtu_files):
        if step_idx % 5 == 0 or step_idx == len(vtu_files)-1:
            print(f" -> Processing step {step_idx}/{len(vtu_files)-1}: {os.path.basename(vtu_path)}")
            
        mesh_vtu = pv.read(vtu_path)
        current_inflow = 0.0
        current_outflow = 0.0

        for name, mask in all_masks.items():
            # 1. Muestrear el volumen en la cara
            cap_results = mask.sample(mesh_vtu)
            
            # 2. Convertir a datos de celda para tener el área real de los elementos
            cap_cells = cap_results.point_data_to_cell_data()
            areas = cap_cells.compute_cell_sizes()['Area']
            vel_vectors = cap_cells.cell_data['Velocity']
            
            # 3. CORRECCIÓN DE LA NORMAL: Usar la orientación global del plano del CAP
            # Calculamos la normal promedio rigurosa del plano para evitar distorsiones locales
            cap_cells_with_normals = cap_cells.compute_normals(cell_normals=True, point_normals=False)
            normals = cap_cells_with_normals.cell_data['Normals']
            mean_normal = np.mean(normals, axis=0)
            mean_normal /= np.linalg.norm(mean_normal) # Vector unitario estable del contorno
            
            # 4. Producto escalar estable para toda la sección transversal
            vel_normal_comp = np.dot(vel_vectors, mean_normal)
            face_flow = np.sum(vel_normal_comp * areas) / 1000.0  # mL/s
            
            # 5. Clasificación basada en la dirección del flujo respecto al nombre anatómico
            if name in inlets:
                # En los inlets, el flujo neto entrante debe ser positivo para nuestro balance
                current_inflow += abs(face_flow)
            else:
                # En los outlets, el flujo saliente es positivo
                current_outflow += abs(face_flow)

        if current_inflow > 0:
            error_pct = (abs(current_inflow - current_outflow) / current_inflow) * 100.0
        else:
            error_pct = 0.0

        total_inflow_list.append(current_inflow)
        total_outflow_list.append(current_outflow)
        mass_error_list.append(error_pct)
        times.append(step_idx * time_step)

    return np.array(times), np.array(total_inflow_list), np.array(total_outflow_list), np.array(mass_error_list)

# --- CONFIGURACIÓN DE RUTAS ---
surfaces_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Patient_models/pacs-scd-005/Meshes/Geometric_UQ_PACS005/Variation_0/Simulation_file/mesh-complete/mesh-surfaces/"
simulation_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Patient_models/pacs-scd-005/Meshes/Geometric_UQ_PACS005/Variation_0/Simulation_file/96-procs/"

surfaces_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Patient_models/pacs-scd-005/Simulations/fine/5_asl/mesh-complete/mesh-surfaces/"
simulation_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Patient_models/pacs-scd-005/Simulations/fine/5_asl/96-procs/"

inlets = ["L_ICA", "R_ICA", "L_VA", "R_VA"]
outlets = ["L_SCA", "R_SCA", "L_PCA", "R_PCA", "L_MCA", "R_MCA", "L_ACA", "R_ACA"]
outlets = ["L_SCA", "R_SCA", "L_PCA", "R_PCA", "L_ICA_2", "R_ICA_2", "L_ACA", "R_ACA", "R_VA_2"]

times, total_inlet, total_outlet, mass_error = check_mass_continuity_perfect(
    vtu_folder=simulation_file, surfaces_dir=surfaces_dir,
    inlets=inlets, outlets=outlets, time_step_size=0.001022, saved_every=20
)

# --- PLOTEO ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax1.plot(times, total_inlet, label="Total Inlet Flow", color="#1f77b4", linewidth=2.5)
ax1.plot(times, total_outlet, label="Total Outlet Flow", color="#ff7f0e", linestyle="--", linewidth=2.5)
ax1.set_title("Corrected Mass Continuity Verification (Mean Normal Projection)", fontsize=14)
ax1.set_ylabel("Flow (mL/s)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

ax2.plot(times, mass_error, color="#d62728", linewidth=1.5)
ax2.set_xlabel("Time (s)", fontsize=12)
ax2.set_ylabel("Relative Error (%)", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"New Real Average Mass Continuity Error: {np.mean(mass_error):.4f}%")