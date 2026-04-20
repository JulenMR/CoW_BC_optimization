import pysvzerod
import json
import pandas as pd
import importlib.util
import sys
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

json_file_original = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Models/zeroD_simulation/zeroD_script.json"
json_file_1_phase = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script_phase1.json"
json_file_optimized = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script_optimized_prueba.json"
save_path = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/"
model_config = json.load(open(json_file_optimized))

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

plot_custom_0d_results(df, inlets, "flow_in", BRANCH_MAPPING, title="PACS008 0D Flow results")
plot_custom_0d_results(df, inlets, "pressure_in", BRANCH_MAPPING, title="PACS008 0D Pressure results")

plot_custom_0d_results(df, outlets, "flow_out", BRANCH_MAPPING, title="PACS008 0D Flow results")
plot_custom_0d_results(df, outlets, "pressure_out", BRANCH_MAPPING, title="PACS008 0D Pressure results")




