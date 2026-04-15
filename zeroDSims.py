import pysvzerod
import json
import pandas as pd
import importlib.util
import sys
import os
import matplotlib.pyplot as plt

json_file_original = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script.json"
json_file_1_phase = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Models/zeroD_simulation/zeroD_script_optimized.json"
json_file_optimized = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Models/zeroD_simulation/zeroD_script_phase1.json"

model_config = json.load(open(json_file_original))

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
            
            if "pressure" in parameter:
                y_values = y_values / 1333.3
            elif "flow" in parameter:
                y_values = y_values / 1000

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

def check_mass_conservation(df, inlet_names, outlet_names, period=1.022):
    t_max = df['time'].max()
    last_cycle = df[df['time'] > (t_max - period)]
    
    total_inflow = 0
    print("\n>>> (Inlets):")
    for name in inlet_names:
        avg_q = last_cycle[last_cycle['name'] == name]['flow_in'].mean()
        avg_q = abs(avg_q)
        print(f"  {name}: {avg_q:8.2f} mm3/s")
        total_inflow += avg_q
        
    total_outflow = 0
    print("\n>>> (Outlets):")
    for name in outlet_names:
        avg_q = last_cycle[last_cycle['name'] == name]['flow_out'].mean()
        print(f"  {name}: {avg_q:8.2f} mL/s")
        total_outflow += avg_q
        
    difference = abs(total_inflow - total_outflow)
    error_relativo = (difference / total_inflow) * 100
    
    print("\n" + "="*40)
    print(f"TOTAL INFLOW:  {total_inflow:10.2f} mL/s")
    print(f"TOTAL OUTFLOW: {total_outflow:10.2f} mL/s")
    print(f"DIFERENCIA:    {difference:10.4f} mL/s")
    print(f"ERROR:         {error_relativo:10.6f} %")
    print("="*40)

check_mass_conservation(df, inlets, outlets)

p_start = df[df['name'] == 'branch0']['pressure_in'].iloc[-1022]
p_end = df[df['name'] == 'branch0']['pressure_in'].iloc[-1]
print(f"Difference between inlet and outlet: {p_end - p_start}")

plot_custom_0d_results(df, inlets, "flow_in" )
plot_custom_0d_results(df, inlets, "flow_out")

plot_custom_0d_results(df, inlets, "pressure_in" )
plot_custom_0d_results(df, inlets, "pressure_out")




