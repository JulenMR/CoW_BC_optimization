import pysvzerod
import json
import pandas as pd
import importlib.util
import sys
import os
import matplotlib.pyplot as plt

json_file_original = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Models/zeroD_simulation/zeroD_script.json"
json_file_1_phase = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script_phase1.json"
json_file_optimized = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Models/zeroD_simulation/zeroD_script_optimized.json"

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

def plot_custom_0d_results(df, branchnames, parameter, mapping):
    plt.figure(figsize=(10, 6))
    found_any = False
    
    is_pressure = "pressure" in parameter
    unit_label = " (mmHg)" if is_pressure else " (mL/s)"

    for name in branchnames:
        branch_data = df[df['name'] == name]
        label_name = mapping[name]
        if not branch_data.empty:
            y_values = branch_data[parameter].copy()
            
            if "pressure" in parameter:
                y_values = y_values / 133.3
            elif "flow" in parameter:
                y_values = y_values / 1000

            plt.plot(
                branch_data['time'], 
                y_values, 
                label=f'{parameter}: {label_name}', 
                linewidth=2
            )
            
            found_any = True

        else:
            print(f"Warning: branch '{name}' not found")

    if not found_any:
        print("Branches not found")
        plt.close()
        return

    plt.title(f'{parameter.replace("_", " ").title()} in selected branches')
    plt.xlabel('Time (s)')
    plt.ylabel(parameter + unit_label)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

inlets = ['branch0', 'branch1', 'branch2', 'branch3'] 
outlets = ['branch4', 'branch5', 'branch6', 'branch7', 'branch8', 'branch9', 'branch10', 'branch11'] 
total = inlets + outlets

BRANCH_MAPPING = {
    "branch0": "ICA_L", "branch1": "ICA_R", "branch2": "VA_L", "branch3": "VA_R", "branch4": "SCA_L", "branch5": "PCA_L", 
    "branch6": "MCA_L", "branch7": "ACA_L", "branch8": "ACA_R", "branch9": "MCA_R", "branch10": "PCA_R", "branch11": "SCA_R"
    }

plot_custom_0d_results(df, outlets, "flow_in", BRANCH_MAPPING)
plot_custom_0d_results(df, outlets, "flow_out", BRANCH_MAPPING)

plot_custom_0d_results(df, outlets, "pressure_in", BRANCH_MAPPING)
plot_custom_0d_results(df, outlets, "pressure_out", BRANCH_MAPPING)




