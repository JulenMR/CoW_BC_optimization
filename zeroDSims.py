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
import time

def plot_combined_0d_3d_results(zeroD_json_filepath, filepath_3D, branchnames, parameter, mapping, title, save_path = None):
    
    # 1. Execute 0D simulation
    zeroD_json = json.load(open(zeroD_json_filepath))
    print("Executing 0D simulation...")
    solver = pysvzerod.Solver(zeroD_json)
    solver.run()
    df_zeroD = solver.get_full_result()
    
    # 2. Load 3D data
    flow_3d_opt_path = os.path.join(filepath_3D, "B_NS_Velocity_flux.txt")
    pressure_3d_opt_path = os.path.join(filepath_3D, "B_NS_Pressure_average.txt")
    
    # Select the 3D file based on the requested parameter
    file_3d = pressure_3d_opt_path if "pressure" in parameter else flow_3d_opt_path
    df_3d = pd.read_csv(file_3d, sep='\s+')
    df_3d_last = df_3d.tail(1000).copy()
    
    fig, (ax0, ax3) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    is_pressure = "pressure" in parameter
    unit_label = " (mmHg)" if is_pressure else " (mL/s)"

    found_any = False

    for name in branchnames:
        label_name = mapping[name]
        line_color = COLOR_MAP.get(label_name, "grey")
        
        # 0D PLOT
        branch_data_0d = df_zeroD[df_zeroD['name'] == name]
        if not branch_data_0d.empty:
            y_0d = branch_data_0d[parameter].values.copy()
            t_0d = branch_data_0d['time'].values
            
            # Unit conversion
            if is_pressure:
                y_0d /= 133.3 
            else:
                y_0d /= 1000.0 

            ax0.plot(t_0d, y_0d, label=label_name, linewidth=2, color=line_color)
            
            # 3D PLOT 
            if label_name in df_3d_last.columns:
                y_3d = df_3d_last[label_name].values.copy()
                
                t_3d = np.linspace(t_0d.min(), t_0d.max(), len(y_3d))
                
                if is_pressure:
                    y_3d /= 133.3 
                else:
                    y_3d /= 1000.0 
                
                ax3.plot(t_3d, np.abs(y_3d), label=label_name, linewidth=2, color=line_color)
            
            found_any = True

    if not found_any:
        print("No branches found to plot.")
        plt.close()
        return

    # 0D Subplot
    ax0.set_title(f"0D Solver: {parameter}")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel(parameter + unit_label)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='upper right', fontsize='small', ncol=2)

    # 3D Subplot
    ax3.set_title(f"3D FSI: {parameter}")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Main figure title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    if save_path:
        output_name = f"combined_0D_3D_{parameter}.png"
        plt.savefig(os.path.join(save_path, output_name), dpi=300)
    plt.show()


def simulation_comparison(zeroD_filepath, optimized_3d_filepath, manual_3d_filepath, clinical_data_filepath, patient_number, save_path = None):

    # Run 0D solver
    zeroD_json = json.load(open(zeroD_filepath))
    print("Executing 0D simulation")
    solver = pysvzerod.Solver(zeroD_json)
    solver.run()
    df_zeroD = solver.get_full_result()

    # Load clinical data
    target_p, target_pulse, clinical_flows = get_clinical_data(file=clinical_data_filepath, p_number=patient_number)

    # Load optimized 3D simulated data
    flow_3d_opt_path = os.path.join(optimized_3d_filepath, "B_NS_Velocity_flux.txt")
    opt_flow_3d = pd.read_csv(flow_3d_opt_path, sep='\s+')
    df_opt_flow_3d_last = opt_flow_3d.tail(1000)

    # Load manual 3D simulated data
    flow_3d_manual_path = os.path.join(manual_3d_filepath, "B_NS_Velocity_flux.txt")
    manual_flow_3d = pd.read_csv(flow_3d_manual_path, sep='\s+')
    df_manual_flow_3d_last = manual_flow_3d.tail(1000)

    print("\nCOMPARISON: Clinical data vs Optimized 0D vs Optimized 3D vs Manual 3D")
    print(f"{'Vessel':<12} | {'Clinical':>10} | {'0D Mean':>10} | {'Opt 3D':>10} | {'Manual 3D':>10} | {'Err manual%':>12} | {'Err optimized%':>12}")
    print("-" * 95)

    full_comparison = []

    for b_id, cap_name in BRANCH_MAPPING.items():
        if b_id in outlets: 
            # 0D Flow
            col_0d = "flow_in" if b_id in inlets else "flow_out"
            mean_0d = df_zeroD[df_zeroD['name'] == b_id][col_0d].mean() / 1000.0

            # 3D Flows (Converting to mL/s)
            mean_3d_opt = np.abs(df_opt_flow_3d_last[cap_name].mean() / 1000.0) if cap_name in df_opt_flow_3d_last.columns else np.nan
            mean_3d_manual = np.abs(df_manual_flow_3d_last[cap_name].mean() / 1000.0) if cap_name in df_manual_flow_3d_last.columns else np.nan
            
            # Clinical Flow
            q_clinical = clinical_flows.get(cap_name, 0.0) / 1000.0
            
            # Error calculations
            zeroD_err = abs(mean_0d - q_clinical) / q_clinical * 100 if q_clinical > 0 else 0
            opt_err = abs(mean_3d_opt - q_clinical) / q_clinical * 100 if q_clinical > 0 else 0
            manual_err = abs(mean_3d_manual - q_clinical) / q_clinical * 100 if q_clinical > 0 else 0

            print(f"{cap_name:<12} | {q_clinical:10.4f} | {mean_0d:10.4f} | {mean_3d_opt:10.4f} | {mean_3d_manual:10.4f} | {manual_err:11.1f}% | {opt_err:11.1f}% ")
            
            full_comparison.append({
                "Vessel": cap_name,
                "Clinical": q_clinical,
                "0D": mean_0d,
                "Opt_3D": mean_3d_opt,
                "Manual_3D": mean_3d_manual,
                "ZeroD_Err": zeroD_err,
                "Opt_Err": opt_err,
                "Manual_Err": manual_err
            })
        else:
            continue

    # Barplot
    df_plot = pd.DataFrame(full_comparison)
    plt.figure(figsize=(16, 7))

    x = np.arange(len(df_plot))
    width = 0.2 

    plt.bar(x - 1.5*width, df_plot['Clinical'],  width, label='Clinical target', color='#2ecc71', alpha=0.8)
    plt.bar(x - 0.5*width, df_plot['0D'],        width, label='0D simulation',  color='#3498db', alpha=0.8)
    plt.bar(x + 0.5*width, df_plot['Opt_3D'],    width, label='Optimized 3D',   color='#e74c3c', alpha=0.8)
    plt.bar(x + 1.5*width, df_plot['Manual_3D'], width, label='Manual 3D',      color='#f1c40f', alpha=0.8)

    for i in range(len(df_plot)):
        if not np.isnan(df_plot['0D'][i]):
            plt.text(x[i] - 0.5*width, df_plot['0D'][i] + 0.01, 
                     f"{df_plot['ZeroD_Err'][i]:.1f}%", 
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=0)
            
        if not np.isnan(df_plot['Opt_3D'][i]):
            plt.text(x[i] + 0.5*width, df_plot['Opt_3D'][i] + 0.01, 
                     f"{df_plot['Opt_Err'][i]:.1f}%", 
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=0)
        
        if not np.isnan(df_plot['Manual_3D'][i]):
            plt.text(x[i] + 1.5*width, df_plot['Manual_3D'][i] + 0.01, 
                     f"{df_plot['Manual_Err'][i]:.1f}%", 
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=0)

    plt.ylabel('Mean Flow (mL/s)', fontsize=12)
    plt.title(f'Clinical vs 0D vs Optimized 3D vs Manual 3D for PACS{patient_number:03d}', fontsize=14)
    plt.xticks(x, df_plot['Vessel'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save_path is not None: 
        plt.savefig(os.path.join(save_path, f"comparison_validation_PACS{patient_number:03d}.png"), dpi=300)
    plt.show()

def check_3d_flow_continuity(optimized_3d_filepath, mapping_dict):
    flow_3d_path = os.path.join(optimized_3d_filepath, "B_NS_Velocity_flux.txt")
    df_3d = pd.read_csv(flow_3d_path, sep='\s+')
    df_last = df_3d.tail(1000)
    
    inlet_names = [k for k, v in mapping_dict.items() if v <= 3]
    outlet_names = [k for k, v in mapping_dict.items() if v > 3]
    
    total_inflow = sum(np.abs(df_last[name].sum()) for name in inlet_names if name in df_last.columns)
    total_outflow = sum(np.abs(df_last[name].sum()) for name in outlet_names if name in df_last.columns)
    
    continuity_error = total_inflow - total_outflow
    percent_error = (np.abs(continuity_error) / total_inflow) * 100 if total_inflow > 0 else 0

    inflows = df_last[inlet_names].abs().sum(axis=1)
    outflows = df_last[outlet_names].abs().sum(axis=1)
    max_instant_error = ((inflows - outflows).abs() / inflows).max() * 100
    print(f"Error instantáneo máximo: {max_instant_error:10.2f} %")

    print("\n" + "="*45)
    print(" Flow balance")
    print("="*45)
    print(f"{'TOTAL INFLOW:':<20} {total_inflow/1000.0:10.4f} mL/s")
    print(f"{'TOTAL OUTFLOW:':<20} {total_outflow/1000.0:10.4f} mL/s")
    print("-" * 45)
    print(f"{'Total diference:':<20} {continuity_error/1000.0:10.4f} mL/s")
    print(f"{'Percentage error:':<20} {percent_error:10.2f} %")
    print("="*45)


import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def check_and_plot_3d_continuity(optimized_3d_filepath, mapping_dict):
    # 1. Carga de datos
    flow_3d_path = os.path.join(optimized_3d_filepath, "B_NS_Velocity_flux.txt")
    if not os.path.exists(flow_3d_path):
        print(f"file not found: {flow_3d_path}")
        return

    df_3d = pd.read_csv(flow_3d_path, sep='\s+')
    
    # Usamos el último ciclo (por ejemplo, últimos 1000 pasos) para el plot
    df_last = df_3d.tail(1000).copy()
    # Resetear índice para que el eje X empiece en 0 o represente el timestep relativo
    df_last = df_last.reset_index(drop=True)

    # 2. Identificar Inlets y Outlets
    inlet_names = [k for k, v in mapping_dict.items() if v <= 3 and k in df_last.columns]
    outlet_names = [k for k, v in mapping_dict.items() if v > 3 and k in df_last.columns]

    # 3. Cálculo de series temporales (mL/s) - Asumiendo entrada en mm3/s
    # Sumamos todos los inlets y outlets por cada fila (timestep)
    series_inflow = df_last[inlet_names].abs().sum(axis=1) / 1000.0
    series_outflow = df_last[outlet_names].abs().sum(axis=1) / 1000.0
    series_error = (series_inflow - series_outflow).abs()

    # 4. Estadísticas globales para la consola
    total_inflow_avg = series_inflow.mean()
    total_outflow_avg = series_outflow.mean()
    percent_error_avg = (abs(total_inflow_avg - total_outflow_avg) / total_inflow_avg) * 100
    
    print("\n" + "="*45)
    print(f"{'TOTAL INFLOW AVG:':<20} {total_inflow_avg:10.4f} mL/s")
    print(f"{'TOTAL OUTFLOW AVG:':<20} {total_outflow_avg:10.4f} mL/s")
    print(f"{'ERROR MEDIO:':<20} {percent_error_avg:10.2f} %")
    print(f"{'ERROR INST. MÁX:':<20} {(series_error / series_inflow).max() * 100:10.2f} %")
    print("="*45)

    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Flujos
    plt.plot(series_inflow, label='Total Inflow', color='royalblue', linewidth=2)
    plt.plot(series_outflow, label='Total Outflow', color='crimson', linestyle='--', linewidth=2)
    plt.ylabel('Flow (mL/s)')
    plt.title('Flow balance for manual PACS005')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

# Ejemplo de uso:
# check_and_plot_3d_continuity(optimized_3d_filepath, mapping_dict)


if __name__ == "__main__":
    patient_number = 5
    zeroD_json_file = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models/zeroD_simulation/zeroD_script_optimized.json"
    opt_3d_result = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Simulations/fine/{patient_number}_zeroD_opt/122-procs"
    manual_result_3d = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Simulations/fine/{patient_number}_asl/96-procs"
    clinical_data_file = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/subject_targets.csv"
    save_path = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Simulations/fine/{patient_number}_zeroD_opt"

    mapping_dict = {
        "L_ICA":0, "R_ICA":1, "L_VA":2, "R_VA":3, "L_SCA":4, "L_PCA":5,
        "L_MCA":6, "L_ACA":7, "R_ACA":8, "R_MCA":9, "R_PCA":10, "R_SCA":11,
    }
    
    BRANCH_MAPPING = {f"branch{v}": k for k, v in mapping_dict.items()}

    inlets = [f"branch{v}" for v in mapping_dict.values() if v <= 3]
    outlets = [f"branch{v}" for v in mapping_dict.values() if v > 3]
    total = inlets + outlets

    colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(mapping_dict)))
    COLOR_MAP = dict(zip(mapping_dict.keys(), colors))

    total_ordered = sorted(total, key=lambda b: mapping_dict[BRANCH_MAPPING[b]])

    #check_and_plot_3d_continuity(manual_result_3d, mapping_dict)

    # simulation_comparison(zeroD_filepath=zeroD_json_file, optimized_3d_filepath=opt_3d_result, manual_3d_filepath=manual_result_3d, 
    #                       patient_number= patient_number, clinical_data_filepath=clinical_data_file, save_path=save_path)
    
    # plot_combined_0d_3d_results(zeroD_json_filepath = zeroD_json_file, filepath_3D = opt_3d_result, branchnames = inlets, 
    #                             parameter = "flow_in", mapping = BRANCH_MAPPING, title = "PACS011 inlet comparison", save_path=None)
    start_time = time.time()

    zeroD_json = json.load(open(zeroD_json_file))
    print("Executing 0D simulation")
    solver = pysvzerod.Solver(zeroD_json)
    solver.run()
    df_zeroD = solver.get_full_result()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")