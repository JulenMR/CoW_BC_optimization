import pysvzerod
import json
import pandas as pd
import importlib.util
import sys
import os
import matplotlib.pyplot as plt

json_file_full = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Models/zeroD_simulation/zeroD_script.json"
model_config = json.load(open(json_file_full))

# Run solver
print("Executing 0D simulation")
solver = pysvzerod.Solver(model_config)
solver.run()

# Obtain df with results
df = solver.get_full_result()
print(df.info())
print(f"Tiempo máximo en la simulación: {df['time'].max()} segundos")
print(f"Número total de filas: {len(df)}")

def plot_custom_0d_results(df, branchnames, parameter):
    plt.figure(figsize=(10, 6))
    found_any = False
    
    is_pressure = "pressure" in parameter
    unit_label = " (mmHg)" if is_pressure else " (mL/s)"

    for name in branchnames:
        branch_data = df[df['name'] == name]
        
        if not branch_data.empty:
            y_values = branch_data[parameter].copy()
            
            if is_pressure:
                y_values = y_values / 133.3

            plt.plot(
                branch_data['time'], 
                y_values, 
                label=f'{parameter}: {name}', 
                linewidth=2
            )
            
            found_any = True
            final_val = y_values.iloc[-1]
            print(f" Graphed {name} - {parameter} final: {final_val:.2f}{unit_label}")
        else:
            print(f"Warning: brancg '{name}' not found")

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

mis_ramas = ['branch0', 'branch1', 'branch2', 'branch3', 'branch6', 'branch7', 'branch8', 'branch9'] 
parameter = "flow_in" 
# plot_custom_0d_results(df, mis_ramas, parameter)

outlet_branch_flow = 0
for i in range(8):
    j = 4 + i
    branch_name = f"branch{j}"
    branch_data = df[df['name'] == branch_name]
    flow = branch_data["flow_out"].sum()
    outlet_branch_flow += flow

print(outlet_branch_flow)

inlet_branch_flow = 0
for i in range(4):
    branch_name = f"branch{i}"
    branch_data = df[df['name'] == branch_name]
    flow = branch_data["flow_out"].sum()
    inlet_branch_flow += flow

print(inlet_branch_flow)
