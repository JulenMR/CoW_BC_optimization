import pysvzerod
import json
import pandas as pd
import importlib.util
import sys
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
from generate_json import get_clinical_data

json_file_original = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Models/zeroD_simulation/zeroD_script.json"
json_file_1_phase = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script_phase1.json"
json_file_optimized = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Models/zeroD_simulation_2/zeroD_script_optimized.json"
save_path = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/"
model_config = json.load(open(json_file_optimized))


result_3d = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Simulations/fine/11_asl/96-procs"
pressure_3d = os.path.join(result_3d, "B_NS_Pressure_average.txt")
flow_3d = os.path.join(result_3d, "B_NS_Velocity_flux.txt")

clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/corrected_subject_targets.csv"
mean_p, pulse, clinical_flows = get_clinical_data(file=clinical_data_file, p_number=11)

# Run solver
print("Executing 0D simulation")
solver = pysvzerod.Solver(model_config)
solver.run()

# Obtain df with results
df = solver.get_full_result()
print(df.info())
print(f"Tiempo máximo en la simulación: {df['time'].max()} segundos")
print(f"Número total de filas: {len(df)}")

cap_names = sorted([
    "L_SCA", "R_SCA", 
    "L_PCA", "R_PCA", 
    "L_MCA", "R_MCA", 
    "L_ACA", "R_ACA", 
    "L_ICA", "R_ICA", 
    "L_VA", "R_VA"
])
color_list = cm.get_cmap('tab10')(np.linspace(0, 1, len(cap_names)))
COLOR_MAP = dict(zip(cap_names, color_list))
DEFAULT_COLOR = "gray"

def plot_custom_0d_results(df, branchnames, parameter, mapping, title):
    plt.figure(figsize=(10, 6))
    found_any = False
    
    is_pressure = "pressure" in parameter
    unit_label = " (mmHg)" if is_pressure else " (mL/s)"

    for name in branchnames:
        label_name = mapping[name]
        line_color = COLOR_MAP.get(label_name, DEFAULT_COLOR)
        branch_data = df[df['name'] == name]
        if not branch_data.empty:
            y_values = branch_data[parameter].copy()
            
            if "pressure" in parameter:
                y_values = y_values / 133.3
            elif "flow" in parameter:
                y_values = y_values / 1000

            plt.plot(
                branch_data['time'], 
                y_values, 
                label=f'{label_name}', 
                linewidth=2,
                color= line_color
            )
            
            found_any = True

        else:
            print(f"Warning: branch '{name}' not found")

    if not found_any:
        print("Branches not found")
        plt.close()
        return

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(parameter + unit_label)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"zeroD_{parameter}_results.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
inlets = ['branch0', 'branch1', 'branch2', 'branch3'] 
outlets = ['branch4', 'branch5', 'branch6', 'branch7', 'branch8', 'branch9', 'branch10', 'branch11'] 
total = inlets + outlets

BRANCH_MAPPING = {
    "branch0": "L_ICA", "branch1": "R_ICA", "branch2": "L_VA", "branch3": "R_VA", "branch4": "L_SCA", "branch5": "L_PCA", 
    "branch6": "L_MCA", "branch7": "L_ACA", "branch8": "R_ACA", "branch9": "R_MCA", "branch10": "R_PCA", "branch11": "R_SCA"
    }

order_idx = {name: i for i, name in enumerate(cap_names)}
total_ordered = sorted(total, key=lambda b: order_idx.get(BRANCH_MAPPING[b], 99))

# plot_custom_0d_results(df, inlets, "flow_in", BRANCH_MAPPING, title="PACS008 0D Flow results")
# plot_custom_0d_results(df, inlets, "pressure_in", BRANCH_MAPPING, title="PACS008 0D Pressure results")

# plot_custom_0d_results(df, outlets, "flow_out", BRANCH_MAPPING, title="PACS008 0D Flow results")
# plot_custom_0d_results(df, outlets, "pressure_out", BRANCH_MAPPING, title="PACS008 0D Pressure results")

# 1. Cargar datos clínicos (usando tu función)
clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/subject_targets.csv"
target_p, target_pulse, clinical_flows = get_clinical_data(file=clinical_data_file, p_number=11)

# 2. Cargar datos de flujo 3D
df_flow_3d = pd.read_csv(flow_3d, sep='\s+')
df_flow_3d_last = df_flow_3d.tail(1000)

print("\n>>> COMPARATIVA TOTAL: CLÍNICA vs 0D vs 3D")
print(f"{'Arteria':<12} | {'Clínica':>10} | {'0D Mean':>10} | {'3D Mean':>10} | {'Err 0D-Cli %':>12}")
print("-" * 75)

full_comparison = []

# Iterar sobre el mapping para unificar nombres
for b_id, cap_name in BRANCH_MAPPING.items():
    col_0d = "flow_in" if b_id in inlets else "flow_out"
    mean_0d = df[df['name'] == b_id][col_0d].mean() / 1000.0

    mean_3d = np.abs(df_flow_3d_last[cap_name].mean()/1000.0) if cap_name in df_flow_3d_last.columns else np.nan

    q_clinical = clinical_flows.get(cap_name, 0.0) / 1000.0
    
    err_0d = abs(mean_0d - q_clinical) / q_clinical * 100 if q_clinical > 0 else 0

    print(f"{cap_name:<12} | {q_clinical:10.4f} | {mean_0d:10.4f} | {mean_3d:10.4f} | {err_0d:11.1f}%")
    
    full_comparison.append({
        "Arteria": cap_name,
        "Clinica": q_clinical,
        "0D": mean_0d,
        "3D": mean_3d
    })

# 3. Gráfico de barras triple
df_plot = pd.DataFrame(full_comparison)
plt.figure(figsize=(14, 6))

x = np.arange(len(df_plot))
width = 0.25

plt.bar(x - width, df_plot['Clinica'], width, label='Target Clínico', color='#2ecc71', alpha=0.8)
plt.bar(x,         df_plot['0D'],      width, label='Simulación 0D', color='#3498db', alpha=0.8)
plt.bar(x + width, df_plot['3D'],      width, label='Simulación 3D (FSI)', color='#e74c3c', alpha=0.8)

plt.ylabel('Flujo Medio (mL/s)', fontsize=12)
plt.title('Validación de Resultados: Datos Clínicos vs 0D vs 3D', fontsize=14)
plt.xticks(x, df_plot['Arteria'], rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
#plt.savefig(os.path.join(save_path, "total_comparison_validation.png"), dpi=300)
plt.show()


